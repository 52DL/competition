from PyQt4.QtGui import QPen
from PyQt4.Qt import Qt
from sloth.items import PointItem


class CustomPointItem8(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(4)
class CustomPointItem16(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(8)
class CustomPointItem24(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(12)

class CustomPointItem32(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(16)
class CustomPointItem40(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(20)
class CustomPointItem48(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(24)
class CustomPointItem56(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(28)
class CustomPointItem64(PointItem):
    # display values of x and y as text inside the rectangle
    #defaultAutoTextKeys = ['x', 'y']

    def __init__(self, *args, **kwargs):
        PointItem.__init__(self, *args, **kwargs)

        # set drawing pen to red with width 2
        #self.setPen(QPen(Qt.red, 20))
        self.setRadius(32)
LABELS = (
    {
        'attributes': {
            'class':      '1',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem8,  # use custom rect item instead of sloth's standard item
        'text':     'Point8',
    },
    {
        'attributes': {
            'class':      '2',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem16,  # use custom rect item instead of sloth's standard item
        'text':     'Point16',
    },
    {
        'attributes': {
            'class':      '3',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem24,  # use custom rect item instead of sloth's standard item
        'text':     'Point24',
    },
    {
        'attributes': {
            'class':      '4',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem32,  # use custom rect item instead of sloth's standard item
        'text':     'Point32',
    },
    {
        'attributes': {
            'class':      '5',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem40,  # use custom rect item instead of sloth's standard item
        'text':     'Point40',
    },
    {
        'attributes': {
            'class':      '6',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem48,  # use custom rect item instead of sloth's standard item
        'text':     'Point48',
    },
    {
        'attributes': {
            'class':      '7',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem56,  # use custom rect item instead of sloth's standard item
        'text':     'Point56',
    },
    {
        'attributes': {
            'class':      '8',
        },
        'inserter': 'sloth.items.PointItemInserter',
        'item':     CustomPointItem64,  # use custom rect item instead of sloth's standard item
        'text':     'Point64',
    },
)

