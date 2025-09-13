// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from my_interfaces:msg/SingleTargetEstimate.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/single_target_estimate__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
my_interfaces__msg__SingleTargetEstimate__init(my_interfaces__msg__SingleTargetEstimate * msg)
{
  if (!msg) {
    return false;
  }
  // target_id
  msg->target_id = -1l;
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    my_interfaces__msg__SingleTargetEstimate__fini(msg);
    return false;
  }
  // covariance
  msg->covariance[0] = 0.0l;
  msg->covariance[1] = 0.0l;
  msg->covariance[2] = 0.0l;
  msg->covariance[3] = 0.0l;
  msg->covariance[4] = 0.0l;
  msg->covariance[5] = 0.0l;
  msg->covariance[6] = 0.0l;
  msg->covariance[7] = 0.0l;
  msg->covariance[8] = 0.0l;
  return true;
}

void
my_interfaces__msg__SingleTargetEstimate__fini(my_interfaces__msg__SingleTargetEstimate * msg)
{
  if (!msg) {
    return;
  }
  // target_id
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // covariance
}

bool
my_interfaces__msg__SingleTargetEstimate__are_equal(const my_interfaces__msg__SingleTargetEstimate * lhs, const my_interfaces__msg__SingleTargetEstimate * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // target_id
  if (lhs->target_id != rhs->target_id) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // covariance
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->covariance[i] != rhs->covariance[i]) {
      return false;
    }
  }
  return true;
}

bool
my_interfaces__msg__SingleTargetEstimate__copy(
  const my_interfaces__msg__SingleTargetEstimate * input,
  my_interfaces__msg__SingleTargetEstimate * output)
{
  if (!input || !output) {
    return false;
  }
  // target_id
  output->target_id = input->target_id;
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // covariance
  for (size_t i = 0; i < 9; ++i) {
    output->covariance[i] = input->covariance[i];
  }
  return true;
}

my_interfaces__msg__SingleTargetEstimate *
my_interfaces__msg__SingleTargetEstimate__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__SingleTargetEstimate * msg = (my_interfaces__msg__SingleTargetEstimate *)allocator.allocate(sizeof(my_interfaces__msg__SingleTargetEstimate), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(my_interfaces__msg__SingleTargetEstimate));
  bool success = my_interfaces__msg__SingleTargetEstimate__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
my_interfaces__msg__SingleTargetEstimate__destroy(my_interfaces__msg__SingleTargetEstimate * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    my_interfaces__msg__SingleTargetEstimate__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
my_interfaces__msg__SingleTargetEstimate__Sequence__init(my_interfaces__msg__SingleTargetEstimate__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__SingleTargetEstimate * data = NULL;

  if (size) {
    data = (my_interfaces__msg__SingleTargetEstimate *)allocator.zero_allocate(size, sizeof(my_interfaces__msg__SingleTargetEstimate), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = my_interfaces__msg__SingleTargetEstimate__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        my_interfaces__msg__SingleTargetEstimate__fini(&data[i - 1]);
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
my_interfaces__msg__SingleTargetEstimate__Sequence__fini(my_interfaces__msg__SingleTargetEstimate__Sequence * array)
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
      my_interfaces__msg__SingleTargetEstimate__fini(&array->data[i]);
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

my_interfaces__msg__SingleTargetEstimate__Sequence *
my_interfaces__msg__SingleTargetEstimate__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__SingleTargetEstimate__Sequence * array = (my_interfaces__msg__SingleTargetEstimate__Sequence *)allocator.allocate(sizeof(my_interfaces__msg__SingleTargetEstimate__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = my_interfaces__msg__SingleTargetEstimate__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
my_interfaces__msg__SingleTargetEstimate__Sequence__destroy(my_interfaces__msg__SingleTargetEstimate__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    my_interfaces__msg__SingleTargetEstimate__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
my_interfaces__msg__SingleTargetEstimate__Sequence__are_equal(const my_interfaces__msg__SingleTargetEstimate__Sequence * lhs, const my_interfaces__msg__SingleTargetEstimate__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!my_interfaces__msg__SingleTargetEstimate__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
my_interfaces__msg__SingleTargetEstimate__Sequence__copy(
  const my_interfaces__msg__SingleTargetEstimate__Sequence * input,
  my_interfaces__msg__SingleTargetEstimate__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(my_interfaces__msg__SingleTargetEstimate);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    my_interfaces__msg__SingleTargetEstimate * data =
      (my_interfaces__msg__SingleTargetEstimate *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!my_interfaces__msg__SingleTargetEstimate__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          my_interfaces__msg__SingleTargetEstimate__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!my_interfaces__msg__SingleTargetEstimate__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
