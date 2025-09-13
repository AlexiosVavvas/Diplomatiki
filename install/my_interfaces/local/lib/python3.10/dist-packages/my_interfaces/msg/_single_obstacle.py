# generated from rosidl_generator_py/resource/_idl.py.em
# with input from my_interfaces:msg/SingleObstacle.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'dimensions'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SingleObstacle(type):
    """Metaclass of message 'SingleObstacle'."""

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
                'my_interfaces.msg.SingleObstacle')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__single_obstacle
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__single_obstacle
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__single_obstacle
            cls._TYPE_SUPPORT = module.type_support_msg__msg__single_obstacle
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__single_obstacle

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


class SingleObstacle(metaclass=Metaclass_SingleObstacle):
    """Message class 'SingleObstacle'."""

    __slots__ = [
        '_obs_type',
        '_obs_name',
        '_position',
        '_dimensions',
        '_kappa',
        '_rho0',
    ]

    _fields_and_field_types = {
        'obs_type': 'string',
        'obs_name': 'string',
        'position': 'geometry_msgs/Point',
        'dimensions': 'sequence<double>',
        'kappa': 'double',
        'rho0': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.obs_type = kwargs.get('obs_type', str())
        self.obs_name = kwargs.get('obs_name', str())
        from geometry_msgs.msg import Point
        self.position = kwargs.get('position', Point())
        self.dimensions = array.array('d', kwargs.get('dimensions', []))
        self.kappa = kwargs.get('kappa', float())
        self.rho0 = kwargs.get('rho0', float())

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
        if self.obs_type != other.obs_type:
            return False
        if self.obs_name != other.obs_name:
            return False
        if self.position != other.position:
            return False
        if self.dimensions != other.dimensions:
            return False
        if self.kappa != other.kappa:
            return False
        if self.rho0 != other.rho0:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def obs_type(self):
        """Message field 'obs_type'."""
        return self._obs_type

    @obs_type.setter
    def obs_type(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'obs_type' field must be of type 'str'"
        self._obs_type = value

    @builtins.property
    def obs_name(self):
        """Message field 'obs_name'."""
        return self._obs_name

    @obs_name.setter
    def obs_name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'obs_name' field must be of type 'str'"
        self._obs_name = value

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

    @builtins.property
    def dimensions(self):
        """Message field 'dimensions'."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'dimensions' array.array() must have the type code of 'd'"
            self._dimensions = value
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
                "The 'dimensions' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._dimensions = array.array('d', value)

    @builtins.property
    def kappa(self):
        """Message field 'kappa'."""
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'kappa' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'kappa' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._kappa = value

    @builtins.property
    def rho0(self):
        """Message field 'rho0'."""
        return self._rho0

    @rho0.setter
    def rho0(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rho0' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'rho0' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._rho0 = value
