# 这里我们对FP-Growth算法流程做一个归纳。FP-Growth算法包括以下几步：
# 1）扫描数据，得到所有频繁1项集的的计数。然后删除支持度低于阈值的项，将1项频繁集放入项头表，并按照支持度降序排列。
# 2）扫描数据，将读到的原始数据剔除非频繁1项集，并按照支持度降序排列。
# 3）读入排序后的数据集，插入FP树，插入时按照排序后的顺序，插入FP树中，排序靠前的节点是祖先节点，而靠后的是子孙节点。如果有共用的祖先，则对应的公用祖先节点计数加1。插入后，如果有新节点出现，则项头表对应的节点会通过节点链表链接上新节点。直到所有的数据都插入到FP树后，FP树的建立完成。
# 4）从项头表的底部项依次向上找到项头表项对应的条件模式基。从条件模式基递归挖掘得到项头表项项的频繁项集。
# 5）如果不限制频繁项集的项数，则返回步骤4所有的频繁项集，否则只返回满足项数要求的频繁项集。
class TreeNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count):
        self.count += count

    def display(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(ind + 1)


def create_tree(dataset, min_support):
    header_table = {}
    for trans in dataset:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + dataset[trans]

    for k in list(header_table.keys()):
        if header_table[k] < min_support:
            del (header_table[k])

    freq_item_set = set(header_table.keys())

    if len(freq_item_set) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]

    root = TreeNode('Null Set', 1, None)

    for trans, count in dataset.items():
        local_d = {}
        for item in trans:
            if item in freq_item_set:
                local_d[item] = header_table[item][0]
        if len(local_d) > 0:
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: p[1], reverse=True)]
            update_tree(ordered_items, root, header_table, count)

    return root, header_table


def update_tree(items, in_tree, header_table, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].increment(count)
    else:
        in_tree.children[items[0]] = TreeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1:], in_tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    while node_to_test.next is not None:
        node_to_test = node_to_test.next
    node_to_test.next = target_node


def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, header_table):
    # first node in linked list
    tree_node = header_table[base_pat][1]
    cond_pats = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[tuple(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.next
    return cond_pats


def mine_tree(in_tree, header_table, min_support, pre_fix, freq_item_list):
    big_l = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]
    for base_pat in big_l:
        new_freq_set = pre_fix.copy()
        new_freq_set.add(base_pat)
        freq_item_list.append(new_freq_set)
        cond_patt_bases = find_prefix_path(base_pat, header_table)
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_support)
        if my_head is not None:
            mine_tree(my_cond_tree, my_head, min_support, new_freq_set, freq_item_list)


def load_dataset():
    # return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    dataset = [
        ['牛奶', '面包'],
        ['面包', '尿布', '啤酒', '鸡蛋'],
        ['牛奶', '尿布', '啤酒', '可乐'],
        ['面包', '牛奶', '尿布', '啤酒'],
        ['面包', '牛奶', '尿布', '可乐']
    ]
    return dataset


def create_init_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


if __name__ == '__main__':
    min_support = 2
    min_conf = 0.5
    dataset = load_dataset()
    init_set = create_init_set(dataset)
    fp_tree, header_table = create_tree(init_set, min_support)
    freq_items = []
    mine_tree(fp_tree, header_table, min_support, set([]), freq_items)
    print(freq_items)
