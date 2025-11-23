# generated from rosidl_generator_py/resource/_idl.py.em
# with input from my_interfaces:msg/JoystickData.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_JoystickData(type):
    """Metaclass of message 'JoystickData'."""

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
                'my_interfaces.msg.JoystickData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__joystick_data
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__joystick_data
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__joystick_data
            cls._TYPE_SUPPORT = module.type_support_msg__msg__joystick_data
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__joystick_data

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class JoystickData(metaclass=Metaclass_JoystickData):
    """Message class 'JoystickData'."""

    __slots__ = [
        '_throttle',
        '_aileron',
        '_elevator',
        '_rudder',
        '_switch_state',
    ]

    _fields_and_field_types = {
        'throttle': 'double',
        'aileron': 'double',
        'elevator': 'double',
        'rudder': 'double',
        'switch_state': 'int32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.throttle = kwargs.get('throttle', float())
        self.aileron = kwargs.get('aileron', float())
        self.elevator = kwargs.get('elevator', float())
        self.rudder = kwargs.get('rudder', float())
        self.switch_state = kwargs.get('switch_state', int())

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
        if self.throttle != other.throttle:
            return False
        if self.aileron != other.aileron:
            return False
        if self.elevator != other.elevator:
            return False
        if self.rudder != other.rudder:
            return False
        if self.switch_state != other.switch_state:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def throttle(self):
        """Message field 'throttle'."""
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'throttle' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'throttle' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._throttle = value

    @builtins.property
    def aileron(self):
        """Message field 'aileron'."""
        return self._aileron

    @aileron.setter
    def aileron(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'aileron' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'aileron' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._aileron = value

    @builtins.property
    def elevator(self):
        """Message field 'elevator'."""
        return self._elevator

    @elevator.setter
    def elevator(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'elevator' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'elevator' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._elevator = value

    @builtins.property
    def rudder(self):
        """Message field 'rudder'."""
        return self._rudder

    @rudder.setter
    def rudder(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rudder' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'rudder' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._rudder = value

    @builtins.property
    def switch_state(self):
        """Message field 'switch_state'."""
        return self._switch_state

    @switch_state.setter
    def switch_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'switch_state' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'switch_state' field must be an integer in [-2147483648, 2147483647]"
        self._switch_state = value
