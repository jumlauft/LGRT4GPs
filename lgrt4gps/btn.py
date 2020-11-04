import binarytree


class BTN(binarytree.Node):
    """
    Binary Tree Node inherits from Node in package binarytree

    .. versionadded:: 0.0.1

    Attributes
    ----------
    _parent : BTN
        parent
    str : str
        string to describe the node

    """

    def __init__(self, parent=None, value=0, **kwargs):
        super().__init__(value, **kwargs)
        self.parent = parent
        self.str = id(self)

    @property
    def parent(self):
        """
        Get parent of current node

        Returns
        -------
        parent : BTN
        """
        return self._parent

    @parent.setter
    def parent(self, node):
        """
        Parent for current node

        Parameters
        ----------
        node : BTN
            node to be set as parent
        """
        if node is None or isinstance(node, BTN):
            self._parent = node
        else:
            raise binarytree.NodeTypeError('parent child must be a BTN instance')


    @property
    def is_leaf(self):
        """
        Returns True if current node is a leaf, False otherwise

        Returns
        -------
        is_leaf : bool
        """
        return self.right is None and self.left is None

    @property
    def is_root(self):
        """
        Returns True if current node is a root, False otherwise

        Returns
        -------
        is_root : bool
        """
        return self.parent is None

    def get_root(self):
        """
        Iteratively finds root of the tree

        Returns
        -------
        root : BTN
            root of the tree
        """
        root = self
        while not root.is_root:
            root = root.parent
        return root

    def check_parent_child(self):
        """
        Checks if all parent relations are set

        Returns
        -------
        correct : bool
            True if relations of all subtrees are correct
            False if there is a missing link
        """
        if self.is_leaf:
            return True
        left_correct, right_correct = True, True
        if self.left is not None:
            if self.left.parent != self:
                left_correct = False
            else:
                left_correct = self.left.check_parent_child()
        if self.right is not None:
            if self.right.parent != self:
                right_correct = False
            else:
                right_correct = self.right.check_parent_child()
        return left_correct and right_correct

    def __str__(self):
        """
        Return the pretty-print string for the binary tree.
        """
        lines = _build_tree_string(self, 0, False, '-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))


def _build_tree_string(root, curr_index, index=False, delimiter='-'):
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
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if index:
        node_repr = '{}{}{}'.format(curr_index, delimiter, root.str)
    else:
        node_repr = str(root.str)

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = \
        _build_tree_string(root.left, 2 * curr_index + 1, index, delimiter)
    r_box, r_box_width, r_root_start, r_root_end = \
        _build_tree_string(root.right, 2 * curr_index + 2, index, delimiter)

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