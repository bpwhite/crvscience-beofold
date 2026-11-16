// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

use fold_proto::{
    PeerEndpoint, RaftAppendEntriesRequest, RaftAppendEntriesResponse, RaftLogCommand,
    RaftLogEntry, RaftVoteRequest, RaftVoteResponse,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

/// Pure-Rust Ongaro-Ousterhout Raft consensus and leader-election state machine for the LAN
/// Beowulf cluster. Supports automatic single-node self-promotion when running standalone on
/// 1 PC and seamless majority-quorum (`floor(N/2) + 1`) failover across `3+` nodes.
pub struct LanRaftState {
    pub self_pubkey: [u8; 32],
    pub self_addr: String,
    pub current_term: u64,
    pub voted_for: Option<[u8; 32]>,
    pub role: RaftRole,
    pub current_leader: Option<PeerEndpoint>,
    pub log: Vec<RaftLogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub match_index: HashMap<[u8; 32], u64>,
    pub next_index: HashMap<[u8; 32], u64>,
    votes_received: HashSet<[u8; 32]>,
    last_leader_contact: Instant,
    election_timeout: Duration,
}

impl LanRaftState {
    pub fn new(self_pubkey: [u8; 32], self_addr: String) -> Self {
        let timeout = randomized_election_timeout(&self_pubkey, 0);
        Self {
            self_pubkey,
            self_addr,
            current_term: 0,
            voted_for: None,
            role: RaftRole::Follower,
            current_leader: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            match_index: HashMap::new(),
            next_index: HashMap::new(),
            votes_received: HashSet::new(),
            last_leader_contact: Instant::now(),
            election_timeout: timeout,
        }
    }

    pub fn last_log_index(&self) -> u64 {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    pub fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    pub fn term_at(&self, index: u64) -> Option<u64> {
        if index == 0 {
            return Some(0);
        }
        self.log
            .get((index - 1) as usize)
            .map(|entry| entry.term)
    }

    pub fn election_timed_out(&self) -> bool {
        self.role != RaftRole::Leader && self.last_leader_contact.elapsed() >= self.election_timeout
    }

    pub fn step_down(&mut self, newer_term: u64, known_leader: Option<PeerEndpoint>) {
        if newer_term > self.current_term {
            self.current_term = newer_term;
            self.voted_for = None;
        }
        self.role = RaftRole::Follower;
        if known_leader.is_some() {
            self.current_leader = known_leader;
            self.last_leader_contact = Instant::now();
        }
        self.votes_received.clear();
        self.election_timeout = randomized_election_timeout(&self.self_pubkey, self.current_term);
    }

    /// Starts a new Raft election term. If `total_cluster_voters == 1` (single PC on LAN),
    /// a vote of `1 / 1` immediately achieves quorum (`majority = 1`) and promotes to `Leader`.
    pub fn start_election(&mut self, total_cluster_voters: usize) -> Option<RaftVoteRequest> {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.self_pubkey);
        self.votes_received.clear();
        self.votes_received.insert(self.self_pubkey);
        self.last_leader_contact = Instant::now();
        self.election_timeout = randomized_election_timeout(&self.self_pubkey, self.current_term);

        let majority = (total_cluster_voters.max(1) / 2) + 1;
        if self.votes_received.len() >= majority {
            self.become_leader(&[]);
            return None;
        }

        Some(RaftVoteRequest {
            term: self.current_term,
            candidate_pubkey: self.self_pubkey,
            candidate_addr: self.self_addr.clone(),
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        })
    }

    pub fn record_vote_response(
        &mut self,
        resp: &RaftVoteResponse,
        peer_pubkeys: &[[u8; 32]],
    ) -> bool {
        if resp.term > self.current_term {
            self.step_down(resp.term, None);
            return false;
        }
        if self.role != RaftRole::Candidate || resp.term != self.current_term {
            return false;
        }
        if resp.vote_granted {
            self.votes_received.insert(resp.voter_pubkey);
        }
        let total_voters = peer_pubkeys.len() + 1;
        let majority = (total_voters / 2) + 1;
        if self.votes_received.len() >= majority {
            self.become_leader(peer_pubkeys);
            return true;
        }
        false
    }

    pub fn become_leader(&mut self, peer_pubkeys: &[[u8; 32]]) {
        self.role = RaftRole::Leader;
        self.current_leader = Some(PeerEndpoint {
            node_pubkey: self.self_pubkey,
            addr: self.self_addr.clone(),
        });
        let next_idx = self.last_log_index() + 1;
        self.next_index.clear();
        self.match_index.clear();
        for &pk in peer_pubkeys {
            self.next_index.insert(pk, next_idx);
            self.match_index.insert(pk, 0);
        }
    }

    /// Handles an incoming `RequestVote` RPC according to Ongaro-Ousterhout Section 5.2 & 5.4.1.
    pub fn handle_vote_request(&mut self, req: &RaftVoteRequest) -> RaftVoteResponse {
        if req.term > self.current_term {
            self.step_down(req.term, None);
        }
        if req.term < self.current_term {
            return RaftVoteResponse {
                term: self.current_term,
                vote_granted: false,
                voter_pubkey: self.self_pubkey,
            };
        }

        let can_vote = match self.voted_for {
            None => true,
            Some(prev) => prev == req.candidate_pubkey,
        };

        let my_last_term = self.last_log_term();
        let my_last_idx = self.last_log_index();
        let log_is_up_to_date = req.last_log_term > my_last_term
            || (req.last_log_term == my_last_term && req.last_log_index >= my_last_idx);

        let granted = can_vote && log_is_up_to_date;
        if granted {
            self.voted_for = Some(req.candidate_pubkey);
            self.last_leader_contact = Instant::now();
        }

        RaftVoteResponse {
            term: self.current_term,
            vote_granted: granted,
            voter_pubkey: self.self_pubkey,
        }
    }

    /// Appends a state-machine command on the active Leader and returns its new log index.
    pub fn append_leader_command(
        &mut self,
        command: RaftLogCommand,
        peer_count: usize,
    ) -> Option<u64> {
        if self.role != RaftRole::Leader {
            return None;
        }
        let idx = self.last_log_index() + 1;
        self.log.push(RaftLogEntry {
            index: idx,
            term: self.current_term,
            command,
        });
        if peer_count == 0 {
            self.commit_index = idx;
        }
        Some(idx)
    }

    /// Builds a per-follower `AppendEntries` RPC batch (or empty heartbeat).
    pub fn build_append_entries_for(&self, follower_pubkey: &[u8; 32]) -> RaftAppendEntriesRequest {
        let next_idx = self
            .next_index
            .get(follower_pubkey)
            .copied()
            .unwrap_or_else(|| self.last_log_index() + 1)
            .max(1);
        let prev_log_index = next_idx - 1;
        let prev_log_term = self.term_at(prev_log_index).unwrap_or(0);

        let entries = if (next_idx as usize) <= self.log.len() {
            self.log[(next_idx as usize - 1)..].to_vec()
        } else {
            Vec::new()
        };

        RaftAppendEntriesRequest {
            term: self.current_term,
            leader_pubkey: self.self_pubkey,
            leader_addr: self.self_addr.clone(),
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.commit_index,
        }
    }

    /// Processes an incoming `AppendEntries` RPC on a Follower/Candidate and returns newly
    /// committed `RaftLogCommand`s ready to apply to the local `FoldStore`.
    pub fn handle_append_entries(
        &mut self,
        req: RaftAppendEntriesRequest,
    ) -> (RaftAppendEntriesResponse, Vec<RaftLogCommand>) {
        if req.term < self.current_term {
            return (
                RaftAppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.commit_index,
                    follower_pubkey: self.self_pubkey,
                },
                Vec::new(),
            );
        }

        let leader_ep = PeerEndpoint {
            node_pubkey: req.leader_pubkey,
            addr: req.leader_addr,
        };
        self.step_down(req.term, Some(leader_ep));
        self.last_leader_contact = Instant::now();

        let Some(local_prev_term) = self.term_at(req.prev_log_index) else {
            return (
                RaftAppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_log_index(),
                    follower_pubkey: self.self_pubkey,
                },
                Vec::new(),
            );
        };

        if local_prev_term != req.prev_log_term {
            // Truncate conflicting suffix
            self.log.truncate(req.prev_log_index.saturating_sub(1) as usize);
            return (
                RaftAppendEntriesResponse {
                    term: self.current_term,
                    success: false,
                    match_index: self.last_log_index(),
                    follower_pubkey: self.self_pubkey,
                },
                Vec::new(),
            );
        }

        for entry in req.entries {
            let pos = (entry.index - 1) as usize;
            if pos < self.log.len() {
                if self.log[pos].term != entry.term {
                    self.log.truncate(pos);
                    self.log.push(entry);
                }
            } else {
                self.log.push(entry);
            }
        }

        if req.leader_commit > self.commit_index {
            self.commit_index = req.leader_commit.min(self.last_log_index());
        }

        let newly_committed = self.drain_unapplied_commits();
        (
            RaftAppendEntriesResponse {
                term: self.current_term,
                success: true,
                match_index: self.last_log_index(),
                follower_pubkey: self.self_pubkey,
            },
            newly_committed,
        )
    }

    pub fn record_append_response(
        &mut self,
        resp: &RaftAppendEntriesResponse,
        peer_pubkeys: &[[u8; 32]],
    ) {
        if resp.term > self.current_term {
            self.step_down(resp.term, None);
            return;
        }
        if self.role != RaftRole::Leader || resp.term != self.current_term {
            return;
        }
        if resp.success {
            self.match_index.insert(resp.follower_pubkey, resp.match_index);
            self.next_index
                .insert(resp.follower_pubkey, resp.match_index + 1);
            self.advance_leader_commit_index(peer_pubkeys);
        } else {
            let cur = self
                .next_index
                .get(&resp.follower_pubkey)
                .copied()
                .unwrap_or(2);
            self.next_index
                .insert(resp.follower_pubkey, cur.saturating_sub(1).max(1));
        }
    }

    fn advance_leader_commit_index(&mut self, peer_pubkeys: &[[u8; 32]]) {
        let total = peer_pubkeys.len() + 1;
        let majority = (total / 2) + 1;
        for n in ((self.commit_index + 1)..=self.last_log_index()).rev() {
            if self.term_at(n) != Some(self.current_term) {
                continue;
            }
            let mut count = 1usize; // Self
            for pk in peer_pubkeys {
                if self.match_index.get(pk).copied().unwrap_or(0) >= n {
                    count += 1;
                }
            }
            if count >= majority {
                self.commit_index = n;
                break;
            }
        }
    }

    pub fn drain_unapplied_commits(&mut self) -> Vec<RaftLogCommand> {
        let mut out = Vec::new();
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.log.get((self.last_applied - 1) as usize) {
                out.push(entry.command.clone());
            }
        }
        out
    }
}

fn randomized_election_timeout(pubkey: &[u8; 32], term: u64) -> Duration {
    let mut entropy = [0u8; 8];
    if getrandom::getrandom(&mut entropy).is_err() {
        entropy[0..4].copy_from_slice(&pubkey[0..4]);
        entropy[4..8].copy_from_slice(&(term as u32).to_le_bytes());
    }
    let jitter_ms = u64::from_le_bytes(entropy) % 350;
    Duration::from_millis(380 + jitter_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fold_proto::{WorkUnitId, WorkUnitLease};

    #[test]
    fn three_node_raft_election_replication_and_failover() {
        let pk1 = [1u8; 32];
        let pk2 = [2u8; 32];
        let pk3 = [3u8; 32];

        let mut n1 = LanRaftState::new(pk1, "127.0.0.1:7443".into());
        let mut n2 = LanRaftState::new(pk2, "127.0.0.1:7444".into());
        let mut n3 = LanRaftState::new(pk3, "127.0.0.1:7445".into());

        // 1. Node 1 starts election in 3-node cluster
        let vote_req = n1.start_election(3).expect("needs peer votes");
        let v2 = n2.handle_vote_request(&vote_req);
        assert!(v2.vote_granted);
        let became_leader = n1.record_vote_response(&v2, &[pk2, pk3]);
        assert!(became_leader);
        assert_eq!(n1.role, RaftRole::Leader);

        // 2. Leader appends a LeaseIssued entry and replicates to n2 + n3
        let dummy_lease = WorkUnitLease {
            wu_id: WorkUnitId::new(1, 0, 0, 0),
            lease_token: [9u8; 16],
            assigned_node: pk2,
            target_steps: 1000,
            leased_at_unix_sec: 100,
            expires_at_unix_sec: 400,
            input_checkpoint_blake3: [5u8; 32],
        };
        n1.append_leader_command(
            RaftLogCommand::LeaseIssued {
                lease: dummy_lease.clone(),
            },
            2,
        );

        let ae2 = n1.build_append_entries_for(&pk2);
        let (r2, _) = n2.handle_append_entries(ae2);
        n1.record_append_response(&r2, &[pk2, pk3]);
        assert_eq!(n1.commit_index, 1);

        // Second heartbeat propagates commit_index = 1 to followers
        let ae2_commit = n1.build_append_entries_for(&pk2);
        let (_, committed_on_n2) = n2.handle_append_entries(ae2_commit);
        assert_eq!(committed_on_n2.len(), 1);

        let ae3 = n1.build_append_entries_for(&pk3);
        let (r3, _) = n3.handle_append_entries(ae3);
        n1.record_append_response(&r3, &[pk2, pk3]);

        // 3. Simulate Node 1 crash -> Node 2 starts Term 2 election and wins vote from Node 3
        let vote_req_t2 = n2.start_election(3).expect("term 2 vote req");
        assert_eq!(vote_req_t2.term, 2);
        let v3 = n3.handle_vote_request(&vote_req_t2);
        assert!(v3.vote_granted);
        assert!(n2.record_vote_response(&v3, &[pk1, pk3]));
        assert_eq!(n2.role, RaftRole::Leader);
        assert_eq!(n2.last_log_index(), 1);
    }
}
