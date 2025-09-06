// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from my_interfaces:msg/SingleObstacle.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/single_obstacle__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `obs_type`
// Member `obs_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `dimensions`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
my_interfaces__msg__SingleObstacle__init(my_interfaces__msg__SingleObstacle * msg)
{
  if (!msg) {
    return false;
  }
  // obs_type
  if (!rosidl_runtime_c__String__init(&msg->obs_type)) {
    my_interfaces__msg__SingleObstacle__fini(msg);
    return false;
  }
  // obs_name
  if (!rosidl_runtime_c__String__init(&msg->obs_name)) {
    my_interfaces__msg__SingleObstacle__fini(msg);
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    my_interfaces__msg__SingleObstacle__fini(msg);
    return false;
  }
  // dimensions
  if (!rosidl_runtime_c__double__Sequence__init(&msg->dimensions, 0)) {
    my_interfaces__msg__SingleObstacle__fini(msg);
    return false;
  }
  // kappa
  // rho0
  return true;
}

void
my_interfaces__msg__SingleObstacle__fini(my_interfaces__msg__SingleObstacle * msg)
{
  if (!msg) {
    return;
  }
  // obs_type
  rosidl_runtime_c__String__fini(&msg->obs_type);
  // obs_name
  rosidl_runtime_c__String__fini(&msg->obs_name);
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // dimensions
  rosidl_runtime_c__double__Sequence__fini(&msg->dimensions);
  // kappa
  // rho0
}

bool
my_interfaces__msg__SingleObstacle__are_equal(const my_interfaces__msg__SingleObstacle * lhs, const my_interfaces__msg__SingleObstacle * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // obs_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->obs_type), &(rhs->obs_type)))
  {
    return false;
  }
  // obs_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->obs_name), &(rhs->obs_name)))
  {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // dimensions
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->dimensions), &(rhs->dimensions)))
  {
    return false;
  }
  // kappa
  if (lhs->kappa != rhs->kappa) {
    return false;
  }
  // rho0
  if (lhs->rho0 != rhs->rho0) {
    return false;
  }
  return true;
}

bool
my_interfaces__msg__SingleObstacle__copy(
  const my_interfaces__msg__SingleObstacle * input,
  my_interfaces__msg__SingleObstacle * output)
{
  if (!input || !output) {
    return false;
  }
  // obs_type
  if (!rosidl_runtime_c__String__copy(
      &(input->obs_type), &(output->obs_type)))
  {
    return false;
  }
  // obs_name
  if (!rosidl_runtime_c__String__copy(
      &(input->obs_name), &(output->obs_name)))
  {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // dimensions
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->dimensions), &(output->dimensions)))
  {
    return false;
  }
  // kappa
  output->kappa = input->kappa;
  // rho0
  output->rho0 = input->rho0;
  return true;
}

my_interfaces__msg__SingleObstacle *
my_interfaces__msg__SingleObstacle__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__SingleObstacle * msg = (my_interfaces__msg__SingleObstacle *)allocator.allocate(sizeof(my_interfaces__msg__SingleObstacle), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(my_interfaces__msg__SingleObstacle));
  bool success = my_interfaces__msg__SingleObstacle__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
my_interfaces__msg__SingleObstacle__destroy(my_interfaces__msg__SingleObstacle * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    my_interfaces__msg__SingleObstacle__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
my_interfaces__msg__SingleObstacle__Sequence__init(my_interfaces__msg__SingleObstacle__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__SingleObstacle * data = NULL;

  if (size) {
    data = (my_interfaces__msg__SingleObstacle *)allocator.zero_allocate(size, sizeof(my_interfaces__msg__SingleObstacle), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = my_interfaces__msg__SingleObstacle__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        my_interfaces__msg__SingleObstacle__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
my_interfaces__msg__SingleObstacle__Sequence__fini(my_interfaces__msg__SingleObstacle__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      my_interfaces__msg__SingleObstacle__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

my_interfaces__msg__SingleObstacle__Sequence *
my_interfaces__msg__SingleObstacle__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__SingleObstacle__Sequence * array = (my_interfaces__msg__SingleObstacle__Sequence *)allocator.allocate(sizeof(my_interfaces__msg__SingleObstacle__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = my_interfaces__msg__SingleObstacle__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
my_interfaces__msg__SingleObstacle__Sequence__destroy(my_interfaces__msg__SingleObstacle__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    my_interfaces__msg__SingleObstacle__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
my_interfaces__msg__SingleObstacle__Sequence__are_equal(const my_interfaces__msg__SingleObstacle__Sequence * lhs, const my_interfaces__msg__SingleObstacle__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!my_interfaces__msg__SingleObstacle__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
my_interfaces__msg__SingleObstacle__Sequence__copy(
  const my_interfaces__msg__SingleObstacle__Sequence * input,
  my_interfaces__msg__SingleObstacle__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(my_interfaces__msg__SingleObstacle);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    my_interfaces__msg__SingleObstacle * data =
      (my_interfaces__msg__SingleObstacle *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!my_interfaces__msg__SingleObstacle__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          my_interfaces__msg__SingleObstacle__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!my_interfaces__msg__SingleObstacle__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
