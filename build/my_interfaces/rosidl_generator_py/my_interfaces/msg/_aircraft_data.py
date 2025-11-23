# generated from rosidl_generator_py/resource/_idl.py.em
# with input from my_interfaces:msg/AircraftData.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_AircraftData(type):
    """Metaclass of message 'AircraftData'."""

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
                'my_interfaces.msg.AircraftData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__aircraft_data
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__aircraft_data
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__aircraft_data
            cls._TYPE_SUPPORT = module.type_support_msg__msg__aircraft_data
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__aircraft_data

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


class AircraftData(metaclass=Metaclass_AircraftData):
    """Message class 'AircraftData'."""

    __slots__ = [
        '_header',
        '_north',
        '_east',
        '_down',
        '_altitude',
        '_roll',
        '_pitch',
        '_yaw',
        '_roll_deg',
        '_pitch_deg',
        '_yaw_deg',
        '_u_forward',
        '_v_sideways',
        '_w_downward',
        '_airspeed',
        '_v_north',
        '_v_east',
        '_v_down',
        '_climb_rate',
        '_ground_speed',
        '_p_roll_rate',
        '_q_pitch_rate',
        '_r_yaw_rate',
        '_p_deg_s',
        '_q_deg_s',
        '_r_deg_s',
        '_alpha',
        '_beta',
        '_alpha_deg',
        '_beta_deg',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'north': 'double',
        'east': 'double',
        'down': 'double',
        'altitude': 'double',
        'roll': 'double',
        'pitch': 'double',
        'yaw': 'double',
        'roll_deg': 'double',
        'pitch_deg': 'double',
        'yaw_deg': 'double',
        'u_forward': 'double',
        'v_sideways': 'double',
        'w_downward': 'double',
        'airspeed': 'double',
        'v_north': 'double',
        'v_east': 'double',
        'v_down': 'double',
        'climb_rate': 'double',
        'ground_speed': 'double',
        'p_roll_rate': 'double',
        'q_pitch_rate': 'double',
        'r_yaw_rate': 'double',
        'p_deg_s': 'double',
        'q_deg_s': 'double',
        'r_deg_s': 'double',
        'alpha': 'double',
        'beta': 'double',
        'alpha_deg': 'double',
        'beta_deg': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.north = kwargs.get('north', float())
        self.east = kwargs.get('east', float())
        self.down = kwargs.get('down', float())
        self.altitude = kwargs.get('altitude', float())
        self.roll = kwargs.get('roll', float())
        self.pitch = kwargs.get('pitch', float())
        self.yaw = kwargs.get('yaw', float())
        self.roll_deg = kwargs.get('roll_deg', float())
        self.pitch_deg = kwargs.get('pitch_deg', float())
        self.yaw_deg = kwargs.get('yaw_deg', float())
        self.u_forward = kwargs.get('u_forward', float())
        self.v_sideways = kwargs.get('v_sideways', float())
        self.w_downward = kwargs.get('w_downward', float())
        self.airspeed = kwargs.get('airspeed', float())
        self.v_north = kwargs.get('v_north', float())
        self.v_east = kwargs.get('v_east', float())
        self.v_down = kwargs.get('v_down', float())
        self.climb_rate = kwargs.get('climb_rate', float())
        self.ground_speed = kwargs.get('ground_speed', float())
        self.p_roll_rate = kwargs.get('p_roll_rate', float())
        self.q_pitch_rate = kwargs.get('q_pitch_rate', float())
        self.r_yaw_rate = kwargs.get('r_yaw_rate', float())
        self.p_deg_s = kwargs.get('p_deg_s', float())
        self.q_deg_s = kwargs.get('q_deg_s', float())
        self.r_deg_s = kwargs.get('r_deg_s', float())
        self.alpha = kwargs.get('alpha', float())
        self.beta = kwargs.get('beta', float())
        self.alpha_deg = kwargs.get('alpha_deg', float())
        self.beta_deg = kwargs.get('beta_deg', float())

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
        if self.north != other.north:
            return False
        if self.east != other.east:
            return False
        if self.down != other.down:
            return False
        if self.altitude != other.altitude:
            return False
        if self.roll != other.roll:
            return False
        if self.pitch != other.pitch:
            return False
        if self.yaw != other.yaw:
            return False
        if self.roll_deg != other.roll_deg:
            return False
        if self.pitch_deg != other.pitch_deg:
            return False
        if self.yaw_deg != other.yaw_deg:
            return False
        if self.u_forward != other.u_forward:
            return False
        if self.v_sideways != other.v_sideways:
            return False
        if self.w_downward != other.w_downward:
            return False
        if self.airspeed != other.airspeed:
            return False
        if self.v_north != other.v_north:
            return False
        if self.v_east != other.v_east:
            return False
        if self.v_down != other.v_down:
            return False
        if self.climb_rate != other.climb_rate:
            return False
        if self.ground_speed != other.ground_speed:
            return False
        if self.p_roll_rate != other.p_roll_rate:
            return False
        if self.q_pitch_rate != other.q_pitch_rate:
            return False
        if self.r_yaw_rate != other.r_yaw_rate:
            return False
        if self.p_deg_s != other.p_deg_s:
            return False
        if self.q_deg_s != other.q_deg_s:
            return False
        if self.r_deg_s != other.r_deg_s:
            return False
        if self.alpha != other.alpha:
            return False
        if self.beta != other.beta:
            return False
        if self.alpha_deg != other.alpha_deg:
            return False
        if self.beta_deg != other.beta_deg:
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
    def north(self):
        """Message field 'north'."""
        return self._north

    @north.setter
    def north(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'north' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'north' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._north = value

    @builtins.property
    def east(self):
        """Message field 'east'."""
        return self._east

    @east.setter
    def east(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'east' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'east' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._east = value

    @builtins.property
    def down(self):
        """Message field 'down'."""
        return self._down

    @down.setter
    def down(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'down' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'down' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._down = value

    @builtins.property
    def altitude(self):
        """Message field 'altitude'."""
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'altitude' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'altitude' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._altitude = value

    @builtins.property
    def roll(self):
        """Message field 'roll'."""
        return self._roll

    @roll.setter
    def roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'roll' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'roll' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._roll = value

    @builtins.property
    def pitch(self):
        """Message field 'pitch'."""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pitch' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'pitch' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._pitch = value

    @builtins.property
    def yaw(self):
        """Message field 'yaw'."""
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'yaw' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._yaw = value

    @builtins.property
    def roll_deg(self):
        """Message field 'roll_deg'."""
        return self._roll_deg

    @roll_deg.setter
    def roll_deg(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'roll_deg' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'roll_deg' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._roll_deg = value

    @builtins.property
    def pitch_deg(self):
        """Message field 'pitch_deg'."""
        return self._pitch_deg

    @pitch_deg.setter
    def pitch_deg(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pitch_deg' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'pitch_deg' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._pitch_deg = value

    @builtins.property
    def yaw_deg(self):
        """Message field 'yaw_deg'."""
        return self._yaw_deg

    @yaw_deg.setter
    def yaw_deg(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw_deg' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'yaw_deg' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._yaw_deg = value

    @builtins.property
    def u_forward(self):
        """Message field 'u_forward'."""
        return self._u_forward

    @u_forward.setter
    def u_forward(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'u_forward' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'u_forward' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._u_forward = value

    @builtins.property
    def v_sideways(self):
        """Message field 'v_sideways'."""
        return self._v_sideways

    @v_sideways.setter
    def v_sideways(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_sideways' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'v_sideways' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._v_sideways = value

    @builtins.property
    def w_downward(self):
        """Message field 'w_downward'."""
        return self._w_downward

    @w_downward.setter
    def w_downward(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'w_downward' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'w_downward' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._w_downward = value

    @builtins.property
    def airspeed(self):
        """Message field 'airspeed'."""
        return self._airspeed

    @airspeed.setter
    def airspeed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'airspeed' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'airspeed' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._airspeed = value

    @builtins.property
    def v_north(self):
        """Message field 'v_north'."""
        return self._v_north

    @v_north.setter
    def v_north(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_north' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'v_north' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._v_north = value

    @builtins.property
    def v_east(self):
        """Message field 'v_east'."""
        return self._v_east

    @v_east.setter
    def v_east(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_east' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'v_east' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._v_east = value

    @builtins.property
    def v_down(self):
        """Message field 'v_down'."""
        return self._v_down

    @v_down.setter
    def v_down(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'v_down' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'v_down' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._v_down = value

    @builtins.property
    def climb_rate(self):
        """Message field 'climb_rate'."""
        return self._climb_rate

    @climb_rate.setter
    def climb_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'climb_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'climb_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._climb_rate = value

    @builtins.property
    def ground_speed(self):
        """Message field 'ground_speed'."""
        return self._ground_speed

    @ground_speed.setter
    def ground_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ground_speed' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'ground_speed' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._ground_speed = value

    @builtins.property
    def p_roll_rate(self):
        """Message field 'p_roll_rate'."""
        return self._p_roll_rate

    @p_roll_rate.setter
    def p_roll_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'p_roll_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'p_roll_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._p_roll_rate = value

    @builtins.property
    def q_pitch_rate(self):
        """Message field 'q_pitch_rate'."""
        return self._q_pitch_rate

    @q_pitch_rate.setter
    def q_pitch_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'q_pitch_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'q_pitch_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._q_pitch_rate = value

    @builtins.property
    def r_yaw_rate(self):
        """Message field 'r_yaw_rate'."""
        return self._r_yaw_rate

    @r_yaw_rate.setter
    def r_yaw_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'r_yaw_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'r_yaw_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._r_yaw_rate = value

    @builtins.property
    def p_deg_s(self):
        """Message field 'p_deg_s'."""
        return self._p_deg_s

    @p_deg_s.setter
    def p_deg_s(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'p_deg_s' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'p_deg_s' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._p_deg_s = value

    @builtins.property
    def q_deg_s(self):
        """Message field 'q_deg_s'."""
        return self._q_deg_s

    @q_deg_s.setter
    def q_deg_s(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'q_deg_s' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'q_deg_s' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._q_deg_s = value

    @builtins.property
    def r_deg_s(self):
        """Message field 'r_deg_s'."""
        return self._r_deg_s

    @r_deg_s.setter
    def r_deg_s(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'r_deg_s' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'r_deg_s' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._r_deg_s = value

    @builtins.property
    def alpha(self):
        """Message field 'alpha'."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'alpha' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'alpha' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._alpha = value

    @builtins.property
    def beta(self):
        """Message field 'beta'."""
        return self._beta

    @beta.setter
    def beta(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'beta' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'beta' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._beta = value

    @builtins.property
    def alpha_deg(self):
        """Message field 'alpha_deg'."""
        return self._alpha_deg

    @alpha_deg.setter
    def alpha_deg(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'alpha_deg' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'alpha_deg' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._alpha_deg = value

    @builtins.property
    def beta_deg(self):
        """Message field 'beta_deg'."""
        return self._beta_deg

    @beta_deg.setter
    def beta_deg(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'beta_deg' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'beta_deg' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._beta_deg = value
