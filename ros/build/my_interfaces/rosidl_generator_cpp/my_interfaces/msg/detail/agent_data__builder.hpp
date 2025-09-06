// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/AgentData.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__AGENT_DATA__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__AGENT_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/agent_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_AgentData_active_cbf_flag
{
public:
  explicit Init_AgentData_active_cbf_flag(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::AgentData active_cbf_flag(::my_interfaces::msg::AgentData::_active_cbf_flag_type arg)
  {
    msg_.active_cbf_flag = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_ergodic_cost
{
public:
  explicit Init_AgentData_ergodic_cost(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  Init_AgentData_active_cbf_flag ergodic_cost(::my_interfaces::msg::AgentData::_ergodic_cost_type arg)
  {
    msg_.ergodic_cost = std::move(arg);
    return Init_AgentData_active_cbf_flag(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_inputs
{
public:
  explicit Init_AgentData_inputs(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  Init_AgentData_ergodic_cost inputs(::my_interfaces::msg::AgentData::_inputs_type arg)
  {
    msg_.inputs = std::move(arg);
    return Init_AgentData_ergodic_cost(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_states
{
public:
  explicit Init_AgentData_states(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  Init_AgentData_inputs states(::my_interfaces::msg::AgentData::_states_type arg)
  {
    msg_.states = std::move(arg);
    return Init_AgentData_inputs(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_num_of_inputs
{
public:
  explicit Init_AgentData_num_of_inputs(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  Init_AgentData_states num_of_inputs(::my_interfaces::msg::AgentData::_num_of_inputs_type arg)
  {
    msg_.num_of_inputs = std::move(arg);
    return Init_AgentData_states(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_num_of_states
{
public:
  explicit Init_AgentData_num_of_states(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  Init_AgentData_num_of_inputs num_of_states(::my_interfaces::msg::AgentData::_num_of_states_type arg)
  {
    msg_.num_of_states = std::move(arg);
    return Init_AgentData_num_of_inputs(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_simulation_time
{
public:
  explicit Init_AgentData_simulation_time(::my_interfaces::msg::AgentData & msg)
  : msg_(msg)
  {}
  Init_AgentData_num_of_states simulation_time(::my_interfaces::msg::AgentData::_simulation_time_type arg)
  {
    msg_.simulation_time = std::move(arg);
    return Init_AgentData_num_of_states(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

class Init_AgentData_header
{
public:
  Init_AgentData_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AgentData_simulation_time header(::my_interfaces::msg::AgentData::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_AgentData_simulation_time(msg_);
  }

private:
  ::my_interfaces::msg::AgentData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::AgentData>()
{
  return my_interfaces::msg::builder::Init_AgentData_header();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__AGENT_DATA__BUILDER_HPP_
