class Backpack:
    """A Backpack object class. Has a name and a list of contents.
    Attributes:
    name (str): the name of the backpack's owner.
    contents (list): the contents of the backpack.
    """
    def __init__(self, name): # This function is the constructor.
    """Set the name and initialize an empty list of contents.
    Parameters:
    name (str): the name of the backpack's owner.
    """
        self.name = name # Initialize some attributes.
        self.contents = []
class Backpack:
    def put(self, item):
        self.contents.append(item)
    def take(self, item):
        """Remove 'item' from the backpack's list of contents."""
        self.contents.remove(item)
    def dump(self):
        """Remove everything from the backpack's list of contents."""
        self.contents=[]