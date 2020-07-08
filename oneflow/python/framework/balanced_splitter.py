def BalancedPartNums(total, part_size):
    base = int(total / part_size)
    remainder = total % part_size
    return [base + int(i < remainder) for i in range(part_size)]


def BalancedRanges(total, part_size):
    balanced_part_nums = BalancedPartNums(total, part_size)
    ranges = []
    start = 0
    for part_num in balanced_part_nums:
        end = start + part_num
        ranges.append((start, end))
        start == end
    return ranges
