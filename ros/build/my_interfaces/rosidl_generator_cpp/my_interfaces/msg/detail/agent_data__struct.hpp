// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/AgentData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AGENT_DATA__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__AGENT_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__AgentData __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__AgentData __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct AgentData_
{
  using Type = AgentData_<ContainerAllocator>;

  explicit AgentData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->simulation_time = 0.0;
      this->num_of_states = 0;
      this->num_of_inputs = 0;
      this->ergodic_cost = 0.0;
      this->active_cbf_flag = false;
    }
  }

  explicit AgentData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->simulation_time = 0.0;
      this->num_of_states = 0;
      this->num_of_inputs = 0;
      this->ergodic_cost = 0.0;
      this->active_cbf_flag = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _simulation_time_type =
    double;
  _simulation_time_type simulation_time;
  using _num_of_states_type =
    int8_t;
  _num_of_states_type num_of_states;
  using _num_of_inputs_type =
    int8_t;
  _num_of_inputs_type num_of_inputs;
  using _states_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _states_type states;
  using _inputs_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _inputs_type inputs;
  using _ergodic_cost_type =
    double;
  _ergodic_cost_type ergodic_cost;
  using _active_cbf_flag_type =
    bool;
  _active_cbf_flag_type active_cbf_flag;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__simulation_time(
    const double & _arg)
  {
    this->simulation_time = _arg;
    return *this;
  }
  Type & set__num_of_states(
    const int8_t & _arg)
  {
    this->num_of_states = _arg;
    return *this;
  }
  Type & set__num_of_inputs(
    const int8_t & _arg)
  {
    this->num_of_inputs = _arg;
    return *this;
  }
  Type & set__states(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->states = _arg;
    return *this;
  }
  Type & set__inputs(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->inputs = _arg;
    return *this;
  }
  Type & set__ergodic_cost(
    const double & _arg)
  {
    this->ergodic_cost = _arg;
    return *this;
  }
  Type & set__active_cbf_flag(
    const bool & _arg)
  {
    this->active_cbf_flag = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::AgentData_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::AgentData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::AgentData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::AgentData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::AgentData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::AgentData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::AgentData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::AgentData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::AgentData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::AgentData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__AgentData
    std::shared_ptr<my_interfaces::msg::AgentData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__AgentData
    std::shared_ptr<my_interfaces::msg::AgentData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AgentData_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->simulation_time != other.simulation_time) {
      return false;
    }
    if (this->num_of_states != other.num_of_states) {
      return false;
    }
    if (this->num_of_inputs != other.num_of_inputs) {
      return false;
    }
    if (this->states != other.states) {
      return false;
    }
    if (this->inputs != other.inputs) {
      return false;
    }
    if (this->ergodic_cost != other.ergodic_cost) {
      return false;
    }
    if (this->active_cbf_flag != other.active_cbf_flag) {
      return false;
    }
    return true;
  }
  bool operator!=(const AgentData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AgentData_

// alias to use template instance with default allocator
using AgentData =
  my_interfaces::msg::AgentData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__AGENT_DATA__STRUCT_HPP_
