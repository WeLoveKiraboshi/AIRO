import torch
import torch.nn.functional as F

def idx1d_to_indices3d(idx_1d, nRows, nCols):
    """Convert index in a flattened tensor, which was originally of size (C, nRows, nCols), into its 3D version

    Args:
        idx_1d (torch.Tensor): indices in flattened tensor, shape (nIndices)
        nRows (int): height of the original tensor
        nCols (int): width of the original tensor
    Returns:
        tuple[torch.Tensor]: (channel_idx, row_idx, col_idx), each tensor has shape (nIndices)
    """
    idx_1d = torch.tensor([3, 7, 16])
    nRows = 2
    nCols = 3
    col_idx = torch.Tensor([int(i%nCols) for i in idx_1d])  # shape (nIndices) - TODO
    row_idx = torch.Tensor([int((i - nRows*nCols*int(i/(nRows*nCols)))/nCols) for i in idx_1d])  # shape (nIndices) - TODO
    ch_idx = torch.Tensor([int(i/(nRows*nCols)) for i in idx_1d])  # shape (nIndices) - TODO
    return ch_idx, row_idx, col_idx

heatmap = torch.rand(5, 5)
heatmap[[1, 1, 3, 3], [1, 3, 3, 1]] = torch.tensor([2, 3, 4, 5], dtype=torch.float)
expected_result = (
            torch.tensor([0]*4, dtype=torch.long),  # channel index, Note: [0] * 4 == [0, 0, 0, 0]
            torch.tensor([3, 3, 1, 1], dtype=torch.long),  # row index (i.e. ys)
            torch.tensor([1, 3, 3, 1], dtype=torch.long),  # column index (i.e. xs)
            torch.tensor([5, 4, 3, 2], dtype=torch.float)  # class prob (i.e. score)
        )

K=4
heatmaps = heatmap.unsqueeze(0)
assert len(heatmaps.size()) == 3, "Wrong heatmap size, expected (nClasses, H, W) get {}".format(heatmaps.size())
# sort the flatten version of heatmaps into descending order
heat_sorted, indices1d = torch.flatten(heatmaps, start_dim=0).sort(descending=True)  # (nClasses*H*W)
# take the topK peaks
print(indices1d.shape)
indices1d = indices1d #...  # shape (topK) - TODO: keep the first `topK` elements in indices1d
# store class probability (i.e. score) of each peak
score = heat_sorted  # shape (topK) - TODO: keep the first `topK` elements in heat_sorted
# retrieve (channel, y, x) from indices1d by invoking idx1d_to_indices3d
print(heatmaps.shape)
ch, h, w = heatmaps.shape
chs, ys, xs = idx1d_to_indices3d(indices1d, h, w)  # each has shape (topK) - TODO

print('ch  actual = {}, desired = {}'.format(chs, expected_result[0]))
print('xs  actual = {}, desired = {}'.format(xs, expected_result[1]))
print('ys  actual = {}, desired = {}'.format(ys, expected_result[2]))