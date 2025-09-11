# generated from rosidl_generator_py/resource/_idl.py.em
# with input from my_interfaces:msg/AgentData.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'states'
# Member 'inputs'
# Member 'in_range_agents_ids'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_AgentData(type):
    """Metaclass of message 'AgentData'."""

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
                'my_interfaces.msg.AgentData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__agent_data
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__agent_data
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__agent_data
            cls._TYPE_SUPPORT = module.type_support_msg__msg__agent_data
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__agent_data

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class AgentData(metaclass=Metaclass_AgentData):
    """Message class 'AgentData'."""

    __slots__ = [
        '_header',
        '_simulation_time',
        '_delta_t_ts',
        '_num_of_states',
        '_num_of_inputs',
        '_states',
        '_inputs',
        '_ergodic_cost',
        '_active_cbf_flag',
        '_in_range_agents_ids',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'simulation_time': 'double',
        'delta_t_ts': 'double',
        'num_of_states': 'int8',
        'num_of_inputs': 'int8',
        'states': 'sequence<double>',
        'inputs': 'sequence<double>',
        'ergodic_cost': 'double',
        'active_cbf_flag': 'boolean',
        'in_range_agents_ids': 'sequence<int8>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('double')),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int8')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.simulation_time = kwargs.get('simulation_time', float())
        self.delta_t_ts = kwargs.get('delta_t_ts', float())
        self.num_of_states = kwargs.get('num_of_states', int())
        self.num_of_inputs = kwargs.get('num_of_inputs', int())
        self.states = array.array('d', kwargs.get('states', []))
        self.inputs = array.array('d', kwargs.get('inputs', []))
        self.ergodic_cost = kwargs.get('ergodic_cost', float())
        self.active_cbf_flag = kwargs.get('active_cbf_flag', bool())
        self.in_range_agents_ids = array.array('b', kwargs.get('in_range_agents_ids', []))

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
        if self.header != other.header:
            return False
        if self.simulation_time != other.simulation_time:
            return False
        if self.delta_t_ts != other.delta_t_ts:
            return False
        if self.num_of_states != other.num_of_states:
            return False
        if self.num_of_inputs != other.num_of_inputs:
            return False
        if self.states != other.states:
            return False
        if self.inputs != other.inputs:
            return False
        if self.ergodic_cost != other.ergodic_cost:
            return False
        if self.active_cbf_flag != other.active_cbf_flag:
            return False
        if self.in_range_agents_ids != other.in_range_agents_ids:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def simulation_time(self):
        """Message field 'simulation_time'."""
        return self._simulation_time

    @simulation_time.setter
    def simulation_time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'simulation_time' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'simulation_time' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._simulation_time = value

    @builtins.property
    def delta_t_ts(self):
        """Message field 'delta_t_ts'."""
        return self._delta_t_ts

    @delta_t_ts.setter
    def delta_t_ts(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'delta_t_ts' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'delta_t_ts' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._delta_t_ts = value

    @builtins.property
    def num_of_states(self):
        """Message field 'num_of_states'."""
        return self._num_of_states

    @num_of_states.setter
    def num_of_states(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_of_states' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'num_of_states' field must be an integer in [-128, 127]"
        self._num_of_states = value

    @builtins.property
    def num_of_inputs(self):
        """Message field 'num_of_inputs'."""
        return self._num_of_inputs

    @num_of_inputs.setter
    def num_of_inputs(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_of_inputs' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'num_of_inputs' field must be an integer in [-128, 127]"
        self._num_of_inputs = value

    @builtins.property
    def states(self):
        """Message field 'states'."""
        return self._states

    @states.setter
    def states(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'states' array.array() must have the type code of 'd'"
            self._states = value
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
                "The 'states' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._states = array.array('d', value)

    @builtins.property
    def inputs(self):
        """Message field 'inputs'."""
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'd', \
                "The 'inputs' array.array() must have the type code of 'd'"
            self._inputs = value
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
                "The 'inputs' field must be a set or sequence and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._inputs = array.array('d', value)

    @builtins.property
    def ergodic_cost(self):
        """Message field 'ergodic_cost'."""
        return self._ergodic_cost

    @ergodic_cost.setter
    def ergodic_cost(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ergodic_cost' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'ergodic_cost' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._ergodic_cost = value

    @builtins.property
    def active_cbf_flag(self):
        """Message field 'active_cbf_flag'."""
        return self._active_cbf_flag

    @active_cbf_flag.setter
    def active_cbf_flag(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'active_cbf_flag' field must be of type 'bool'"
        self._active_cbf_flag = value

    @builtins.property
    def in_range_agents_ids(self):
        """Message field 'in_range_agents_ids'."""
        return self._in_range_agents_ids

    @in_range_agents_ids.setter
    def in_range_agents_ids(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'b', \
                "The 'in_range_agents_ids' array.array() must have the type code of 'b'"
            self._in_range_agents_ids = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'in_range_agents_ids' field must be a set or sequence and each value of type 'int' and each integer in [-128, 127]"
        self._in_range_agents_ids = array.array('b', value)
