import sys, os
import binarytree
sys.path.append(os.path.abspath('./'))
from lgrt4gps.btn import BTN


def test_init():
    root = BTN()
    root.left = BTN(value=1,parent=root)
    root.right = BTN(value=2)
    assert root.check_parent_child() == False

    try:
        root.right.parent = 'Not a BTN instance'
        raise AssertionError
    except binarytree.exceptions.NodeTypeError:
        pass

    root.right.parent = root

    assert root.check_parent_child() == True
    assert root.is_root
    assert not root.is_leaf
    assert root.left.is_leaf
    assert not root.left.is_root
    assert root.leaf_count == 2
    root.pprint()
    print(root)
