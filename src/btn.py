import functools

def checkNodeClass(func):
    @functools.wraps(func)
    def wrapper_checkNodeClass(self,node):
        if node is not None and not isinstance(node, BTN):
            raise TypeError('node must be a BTN instance')
        return func(self,node)
    return wrapper_checkNodeClass

class BTN:
    def __init__(self):
        self._leftChild = None  # left child
        self._rightChild = None  # right child
        self._parent = None  # parent node
        self.str = id(self)
    def get_leftChild(self):
        return self._leftChild

    def get_rightChild(self):
        return self._rightChild

    @checkNodeClass
    def set_rightChild(self, node):
        self._rightChild = node
        node.set_parent(self)

    @checkNodeClass
    def set_leftChild(self, node):
        self._leftChild = node
        node.set_parent(self)

    def get_parent(self):
        return self._parent

    @checkNodeClass
    def set_parent(self,node):
        self._parent = node

    def is_leaf(self):
        return self._rightChild is None and self._leftChild is None

    def is_root(self):
        return self._parent is None

    def get_root(self):
        n = self
        while not n.is_root():
            n = n.get_parent()
        return n

    def depth(self):
        left_depth = self._leftChild.depth() if self._leftChild else 0
        right_depth = self._rightChild.depth if self._rightChild  else 0
        return max(left_depth, right_depth) + 1

    def __str__(self):
        lines = build_tree_string(self, 0, index=False, delimiter='-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    def num_leaves(self):
        if self._leftChild is None and self._rightChild is None:
            return 1
        count = 0
        if self._leftChild:
            count += self._leftChild.width()
        if self._rightChild:
            count += self._rightChild.width()
        return count


def build_tree_string(node, curr_index, index=False, delimiter='-'):
    """Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

         https://github.com/joowani/binarytree
    """
    if node is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if index:
        node_repr = '{}{}{}'.format(curr_index, delimiter, node.str)
    else:
        node_repr = str(node.str)

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = \
        build_tree_string(node._leftChild,2 * curr_index + 1, index, delimiter)
    r_box, r_box_width, r_root_start, r_root_end = \
        build_tree_string(node._rightChild,2 * curr_index + 2, index, delimiter)

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(' ' * (l_root + 1))
        line1.append('_' * (l_box_width - l_root))
        line2.append(' ' * l_root + '/')
        line2.append(' ' * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(' ' * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append('_' * r_root)
        line1.append(' ' * (r_box_width - r_root + 1))
        line2.append(' ' * r_root + '\\')
        line2.append(' ' * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
        r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end

# root = BTN(leftChild=BTN(rightChild=BTN()),rightChild=BTN()) # index: 0, value: 1
# print(root)