# generated from rosidl_generator_py/resource/_idl.py.em
# with input from my_interfaces:msg/ObsAvoidanceDebug.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'beta'
# Member 'u_safe'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ObsAvoidanceDebug(type):
    """Metaclass of message 'ObsAvoidanceDebug'."""

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
                'my_interfaces.msg.ObsAvoidanceDebug')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__obs_avoidance_debug
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__obs_avoidance_debug
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__obs_avoidance_debug
            cls._TYPE_SUPPORT = module.type_support_msg__msg__obs_avoidance_debug
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__obs_avoidance_debug

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ObsAvoidanceDebug(metaclass=Metaclass_ObsAvoidanceDebug):
    """Message class 'ObsAvoidanceDebug'."""

    __slots__ = [
        '_psi',
        '_hddot',
        '_two_alpha_h_hdot',
        '_alpha2_h',
        '_beta',
        '_u_safe',
    ]

    _fields_and_field_types = {
        'psi': 'double',
        'hddot': 'double',
        'two_alpha_h_hdot': 'double',
        'alpha2_h': 'double',
        'beta': 'sequence<double>',
        'u_safe': 'sequence<double>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.psi = kwargs.get('psi', float())
        self.hddot = kwargs.get('hddot', float())
        self.two_alpha_h_hdot = kwargs.get('two_alpha_h_hdot', float())
        self.alpha2_h = kwargs.get('alpha2_h', float())
        self.beta = array.array('d', kwargs.get('beta', []))
        self.u_safe = array.array('d', kwargs.get('u_safe', []))

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
        if self.psi != other.psi:
            return False
        if self.hddot != other.hddot:
            return False
        if self.two_alpha_h_hdot != other.two_alpha_h_hdot:
            return False
        if self.alpha2_h != other.alpha2_h:
            return False
        if self.beta != other.beta:
            return False
        if self.u_safe != other.u_safe:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def psi(self):
        """Message field 'psi'."""
        return self._psi

    @psi.setter
    def psi(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'psi' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'psi' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._psi = value

    @builtins.property
    def hddot(self):
        """Message field 'hddot'."""
        return self._hddot

    @hddot.setter
    def hddot(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'hddot' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'hddot' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._hddot = value

    @builtins.property
    def two_alpha_h_hdot(self):
        """Message field 'two_alpha_h_hdot'."""
        return self._two_alpha_h_hdot

    @two_alpha_h_hdot.setter
    def two_alpha_h_hdot(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'two_alpha_h_hdot' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'two_alpha_h_hdot' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._two_alpha_h_hdot = value

    @builtins.property
    def alpha2_h(self):
        """Message field 'alpha2_h'."""
        return self._alpha2_h

    @alpha2_h.setter
    def alpha2_h(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'alpha2_h' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'alpha2_h' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._alpha2_h = value

    @builtins.property
    def beta(self):
        """Message field 'beta'."""
        return self._beta

    @beta.setter
    def beta(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'beta' array.array() must have the type code of 'd'"
            self._beta = value
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
                "The 'beta' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._beta = array.array('d', value)

    @builtins.property
    def u_safe(self):
        """Message field 'u_safe'."""
        return self._u_safe

    @u_safe.setter
    def u_safe(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'u_safe' array.array() must have the type code of 'd'"
            self._u_safe = value
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
                "The 'u_safe' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._u_safe = array.array('d', value)
