import functools

def checkNodeClass(func):
    @functools.wraps(func)
    def wrapper_checkNodeClass(self,node):

        return func(self,node)
    return wrapper_checkNodeClass

class BTN:
    """
    Binary Tree Node

    .. versionadded:: 0.0.1

    Attributes
    ----------
    _leftChild : BTN
        left child
    _rightChild : BTN
        right child
    _parent : BTN
        parent
    str : str
        string to describe the node

    """
    def __init__(self):
        self._leftChild = None  # left child
        self._rightChild = None  # right child
        self._parent = None  # parent node
        self.str = id(self)

    @property
    def leftChild(self):
        """
        Returns left child of current node

        Returns
        -------
        _leftChild : BTN
            left child of current node
        """
        return self._leftChild

    @property
    def rightChild(self):
        """
        Returns right child of current node

        Returns
        -------
        _rightchild : BTN
            right child of current node
        """
        return self._rightChild

    @rightChild.setter
    def rightChild(self, node):
        """
        Set ``node`` as right child of current node

        Also adds instance as parent of ``node``

        Parameters
        ----------
        node : BTN
            node to be set as child
        """
        if node is not None and not isinstance(node, BTN):
            raise TypeError('node must be a BTN instance')
        else:
            self._rightChild = node
            node.set_parent(self)

    @leftChild.setter
    def leftChild(self, node):
        """
        Set ``node`` as left child of current node

        Also adds instance as parent of ``node``

        Parameters
        ----------
        node : BTN
            node to be set as child
        """
        if node is not None and not isinstance(node, BTN):
            raise TypeError('node must be a BTN instance')
        else:
            self._leftChild = node
            node.set_parent(self)
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
    def parent(self,node):
        """
        Set parent for current node

        Parameters
        ----------
        node : BTN
            node to be set as parent
        """
        if node is not None and not isinstance(node, BTN):
            raise TypeError('node must be a BTN instance')
        else:
            self._parent = node

    @property
    def is_leaf(self):
        """
        Returns True if current node is a leaf, False otherwise

        Returns
        -------
        is_leaf : bool
        """
        return self._rightChild is None and self._leftChild is None

    @property
    def is_root(self):
        """
        Returns True if current node is a root, False otherwise

        Returns
        -------
        is_root : bool
        """
        return self._parent is None

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
            root = root.get_parent()
        return root

    @property
    def depth(self):
        """
        Recursively computes depth of the tree

        Returns
        -------
        depth : int
        """
        left_depth = self._leftChild.depth if self._leftChild else 0
        right_depth = self._rightChild.depth if self._rightChild  else 0
        return max(left_depth, right_depth) + 1

    def __str__(self):
        """
        Returns printable string to visualize the tree

        Returns
        -------
        tree_string : str
        """
        lines = build_tree_string(self, 0, index=False, delimiter='-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    @property
    def num_leaves(self):
        """
        Recursively computes number of leaves of the tree

        Returns
        -------
        number_of_leaves : int
        """
        if self._leftChild is None and self._rightChild is None:
            return 1
        count = 0
        if self._leftChild:
            count += self._leftChild.width()
        if self._rightChild:
            count += self._rightChild.width()
        return count

    def __len__(self):
        return self.num_leaves


def build_tree_string(node, curr_index, index=False, delimiter='-'):
    """
    Generate string to visualize a binary tree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    https://github.com/joowani/binarytree

    .. versionadded:: 0.0.1

    Parameters
    ----------
    node : BTN
        Root node of the binary tree.
    curr_index : int
        Level index of the current node (root node is 0).
    index : bool, optional
         If set to True, include the level (default: False).
    delimiter : str, optional
        Delimiter between index and value (default: '-').

    Returns
    -------
    new_box : str
        Box of characters visually representing the current subtree
    width : int
        width of the box
    new_root_start : int
        start positions of the repr string of the new root node value.
    new_root_end : int
        end positions of the repr string of the new root node value.
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

