import torch

from subarea import split_assignment_by_balanced_centers


def test_subdivision_uses_parent_area_graph() -> None:
    device = torch.device("cpu")
    line_adjacency = torch.eye(6, device=device, dtype=torch.bool)[None, :, :]
    line_positions = torch.arange(5, device=device)
    line_adjacency[:, line_positions, line_positions + 1] = True
    line_adjacency[:, line_positions + 1, line_positions] = True
    parent_assignment = torch.tensor([[0, 0, 0, 0, 1, 1]], device=device)
    child_assignment = split_assignment_by_balanced_centers(
        adjacency=line_adjacency,
        parent_assignments=(parent_assignment,),
        parent_counts=(2,),
    )
    assert child_assignment.tolist() == [[0, 0, 1, 1, 0, 1]]
    grandchild_assignment = split_assignment_by_balanced_centers(
        adjacency=line_adjacency,
        parent_assignments=(parent_assignment, child_assignment),
        parent_counts=(2, 2),
    )
    assert grandchild_assignment.tolist() == [[0, 1, 0, 1, 0, 0]]

    shortcut_adjacency = torch.eye(5, device=device, dtype=torch.bool)[None, :, :]
    shortcut_edges = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [2, 3],
            [0, 4],
            [2, 4],
        ],
        device=device,
    )
    shortcut_adjacency[:, shortcut_edges[:, 0], shortcut_edges[:, 1]] = True
    shortcut_adjacency[:, shortcut_edges[:, 1], shortcut_edges[:, 0]] = True
    shortcut_parent_assignment = torch.tensor([[0, 0, 0, 0, 1]], device=device)
    shortcut_child_assignment = split_assignment_by_balanced_centers(
        adjacency=shortcut_adjacency,
        parent_assignments=(shortcut_parent_assignment,),
        parent_counts=(2,),
    )
    assert shortcut_child_assignment.tolist() == [[0, 0, 1, 1, 0]]


if __name__ == "__main__":
    test_subdivision_uses_parent_area_graph()
