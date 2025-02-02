import torch


def assert_tensor_lists_equal(list1, list2):
    assert len(list1) == len(list2), f"Lengths do not match: {len(list1)} != {len(list2)}"
    for tensor1, tensor2 in zip(list1, list2):
        assert torch.equal(tensor1, tensor2), f"Tensors do not match: {tensor1} != {tensor2}"
