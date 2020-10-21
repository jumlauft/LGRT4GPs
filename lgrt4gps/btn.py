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
    def __init__(self, left=None, right=None, parent=None):
        super().__init__(value=0, left=left, right=right)
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
    def parent(self,node):
        """
        Parent for current node

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


    def __setattr__(self, attr, obj):
        """Modified version of ``__setattr__`` to allow arbitrary values

        """
        if attr == 'left':
            if obj is not None and not isinstance(obj, binarytree.Node):
                raise binarytree.exceptions.NodeTypeError('left child must be a Node instance')
        elif attr == 'right':
            if obj is not None and not isinstance(obj, binarytree.Node):
                raise binarytree.exceptions.NodeTypeError('right child must be a Node instance')
        elif attr == 'parent':
            if obj is not None and not isinstance(obj, binarytree.Node):
                raise binarytree.exceptions.NodeTypeError('parent must be a Node instance')
        object.__setattr__(self, attr, obj)
