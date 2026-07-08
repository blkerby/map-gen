#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnionFindSnapshot {
    change_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    changes: Vec<UnionFindChange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnionFindChange {
    Union {
        child_root: usize,
        parent_root: usize,
        parent_size_before: usize,
    },
    ResetNode {
        idx: usize,
        parent_before: usize,
        size_before: usize,
    },
}

impl UnionFind {
    pub fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
            changes: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        for (idx, parent) in self.parent.iter_mut().enumerate() {
            *parent = idx;
        }
        self.size.fill(1);
        self.changes.clear();
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    pub fn find(&self, mut idx: usize) -> usize {
        while self.parent[idx] != idx {
            idx = self.parent[idx];
        }
        idx
    }

    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return false;
        }
        if self.size[root_a] > self.size[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }

        self.changes.push(UnionFindChange::Union {
            child_root: root_a,
            parent_root: root_b,
            parent_size_before: self.size[root_b],
        });
        self.parent[root_a] = root_b;
        self.size[root_b] += self.size[root_a];
        true
    }

    pub fn reset_node(&mut self, idx: usize) {
        debug_assert_eq!(self.find(idx), idx);
        debug_assert_eq!(self.size[idx], 1);
        self.changes.push(UnionFindChange::ResetNode {
            idx,
            parent_before: self.parent[idx],
            size_before: self.size[idx],
        });
        self.parent[idx] = idx;
        self.size[idx] = 1;
    }

    pub fn snapshot(&self) -> UnionFindSnapshot {
        UnionFindSnapshot {
            change_len: self.changes.len(),
        }
    }

    pub fn restore(&mut self, snapshot: UnionFindSnapshot) {
        while self.changes.len() > snapshot.change_len {
            match self.changes.pop().unwrap() {
                UnionFindChange::Union {
                    child_root,
                    parent_root,
                    parent_size_before,
                } => {
                    self.parent[child_root] = child_root;
                    self.size[parent_root] = parent_size_before;
                }
                UnionFindChange::ResetNode {
                    idx,
                    parent_before,
                    size_before,
                } => {
                    self.parent[idx] = parent_before;
                    self.size[idx] = size_before;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(uf: &UnionFind) -> Vec<usize> {
        (0..uf.len()).map(|idx| uf.find(idx)).collect()
    }

    #[test]
    fn initializes_nodes_as_singletons() {
        let uf = UnionFind::new(4);

        assert_eq!(uf.len(), 4);
        assert!(!uf.is_empty());
        assert_eq!(roots(&uf), vec![0, 1, 2, 3]);
    }

    #[test]
    fn unions_disjoint_components() {
        let mut uf = UnionFind::new(4);

        assert!(uf.union(0, 1));
        assert!(uf.union(2, 3));
        assert!(uf.union(0, 2));

        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(3), root);
        assert!(!uf.union(1, 3));
    }

    #[test]
    fn restores_to_snapshot() {
        let mut uf = UnionFind::new(5);
        assert!(uf.union(0, 1));
        let snapshot = uf.snapshot();

        assert!(uf.union(2, 3));
        assert!(uf.union(1, 2));
        assert_eq!(uf.find(0), uf.find(3));

        uf.restore(snapshot);

        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(2), uf.find(3));
        assert_eq!(uf.find(4), 4);
    }

    #[test]
    fn reset_node_restores_with_snapshot() {
        let mut uf = UnionFind::new(3);
        let snapshot = uf.snapshot();

        uf.reset_node(1);
        assert!(uf.union(1, 2));
        assert_eq!(uf.find(1), uf.find(2));

        uf.restore(snapshot);

        assert_eq!(roots(&uf), vec![0, 1, 2]);
    }

    #[test]
    fn clear_resets_nodes_and_change_log() {
        let mut uf = UnionFind::new(3);
        assert!(uf.union(0, 1));
        let snapshot = uf.snapshot();
        assert!(uf.union(1, 2));

        uf.clear();
        assert_eq!(roots(&uf), vec![0, 1, 2]);

        uf.restore(snapshot);
        assert_eq!(roots(&uf), vec![0, 1, 2]);
    }
}
