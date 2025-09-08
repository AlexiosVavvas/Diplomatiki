# generated from rosidl_generator_py/resource/_idl.py.em
# with input from my_interfaces:msg/CkTable.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'ck_values'
# Member 'ck_values_average_in_range'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CkTable(type):
    """Metaclass of message 'CkTable'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('my_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'my_interfaces.msg.CkTable')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__ck_table
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__ck_table
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__ck_table
            cls._TYPE_SUPPORT = module.type_support_msg__msg__ck_table
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__ck_table

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CkTable(metaclass=Metaclass_CkTable):
    """Message class 'CkTable'."""

    __slots__ = [
        '_table_size',
        '_ck_values',
        '_ck_values_average_in_range',
        '_total_erg_cost',
        '_total_erg_cost_in_range',
        '_erg_cost_reduction_perc',
        '_position',
    ]

    _fields_and_field_types = {
        'table_size': 'int32',
        'ck_values': 'sequence<double>',
        'ck_values_average_in_range': 'sequence<double>',
        'total_erg_cost': 'double',
        'total_erg_cost_in_range': 'double',
        'erg_cost_reduction_perc': 'double',
        'position': 'geometry_msgs/Point',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.table_size = kwargs.get('table_size', int())
        self.ck_values = array.array('d', kwargs.get('ck_values', []))
        self.ck_values_average_in_range = array.array('d', kwargs.get('ck_values_average_in_range', []))
        self.total_erg_cost = kwargs.get('total_erg_cost', float())
        self.total_erg_cost_in_range = kwargs.get('total_erg_cost_in_range', float())
        self.erg_cost_reduction_perc = kwargs.get('erg_cost_reduction_perc', float())
        from geometry_msgs.msg import Point
        self.position = kwargs.get('position', Point())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.table_size != other.table_size:
            return False
        if self.ck_values != other.ck_values:
            return False
        if self.ck_values_average_in_range != other.ck_values_average_in_range:
            return False
        if self.total_erg_cost != other.total_erg_cost:
            return False
        if self.total_erg_cost_in_range != other.total_erg_cost_in_range:
            return False
        if self.erg_cost_reduction_perc != other.erg_cost_reduction_perc:
            return False
        if self.position != other.position:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def table_size(self):
        """Message field 'table_size'."""
        return self._table_size

    @table_size.setter
    def table_size(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'table_size' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'table_size' field must be an integer in [-2147483648, 2147483647]"
        self._table_size = value

    @builtins.property
    def ck_values(self):
        """Message field 'ck_values'."""
        return self._ck_values

    @ck_values.setter
    def ck_values(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'ck_values' array.array() must have the type code of 'd'"
            self._ck_values = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'ck_values' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._ck_values = array.array('d', value)

    @builtins.property
    def ck_values_average_in_range(self):
        """Message field 'ck_values_average_in_range'."""
        return self._ck_values_average_in_range

    @ck_values_average_in_range.setter
    def ck_values_average_in_range(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'ck_values_average_in_range' array.array() must have the type code of 'd'"
            self._ck_values_average_in_range = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'ck_values_average_in_range' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._ck_values_average_in_range = array.array('d', value)

    @builtins.property
    def total_erg_cost(self):
        """Message field 'total_erg_cost'."""
        return self._total_erg_cost

    @total_erg_cost.setter
    def total_erg_cost(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'total_erg_cost' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'total_erg_cost' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._total_erg_cost = value

    @builtins.property
    def total_erg_cost_in_range(self):
        """Message field 'total_erg_cost_in_range'."""
        return self._total_erg_cost_in_range

    @total_erg_cost_in_range.setter
    def total_erg_cost_in_range(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'total_erg_cost_in_range' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'total_erg_cost_in_range' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._total_erg_cost_in_range = value

    @builtins.property
    def erg_cost_reduction_perc(self):
        """Message field 'erg_cost_reduction_perc'."""
        return self._erg_cost_reduction_perc

    @erg_cost_reduction_perc.setter
    def erg_cost_reduction_perc(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'erg_cost_reduction_perc' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'erg_cost_reduction_perc' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._erg_cost_reduction_perc = value

    @builtins.property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'position' field must be a sub message of type 'Point'"
        self._position = value
