// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/aircraft_data__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
my_interfaces__msg__AircraftData__init(my_interfaces__msg__AircraftData * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    my_interfaces__msg__AircraftData__fini(msg);
    return false;
  }
  // north
  // east
  // down
  // altitude
  // roll
  // pitch
  // yaw
  // roll_deg
  // pitch_deg
  // yaw_deg
  // u_forward
  // v_sideways
  // w_downward
  // airspeed
  // v_north
  // v_east
  // v_down
  // climb_rate
  // ground_speed
  // p_roll_rate
  // q_pitch_rate
  // r_yaw_rate
  // p_deg_s
  // q_deg_s
  // r_deg_s
  // alpha
  // beta
  // alpha_deg
  // beta_deg
  return true;
}

void
my_interfaces__msg__AircraftData__fini(my_interfaces__msg__AircraftData * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // north
  // east
  // down
  // altitude
  // roll
  // pitch
  // yaw
  // roll_deg
  // pitch_deg
  // yaw_deg
  // u_forward
  // v_sideways
  // w_downward
  // airspeed
  // v_north
  // v_east
  // v_down
  // climb_rate
  // ground_speed
  // p_roll_rate
  // q_pitch_rate
  // r_yaw_rate
  // p_deg_s
  // q_deg_s
  // r_deg_s
  // alpha
  // beta
  // alpha_deg
  // beta_deg
}

bool
my_interfaces__msg__AircraftData__are_equal(const my_interfaces__msg__AircraftData * lhs, const my_interfaces__msg__AircraftData * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // north
  if (lhs->north != rhs->north) {
    return false;
  }
  // east
  if (lhs->east != rhs->east) {
    return false;
  }
  // down
  if (lhs->down != rhs->down) {
    return false;
  }
  // altitude
  if (lhs->altitude != rhs->altitude) {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // pitch
  if (lhs->pitch != rhs->pitch) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  // roll_deg
  if (lhs->roll_deg != rhs->roll_deg) {
    return false;
  }
  // pitch_deg
  if (lhs->pitch_deg != rhs->pitch_deg) {
    return false;
  }
  // yaw_deg
  if (lhs->yaw_deg != rhs->yaw_deg) {
    return false;
  }
  // u_forward
  if (lhs->u_forward != rhs->u_forward) {
    return false;
  }
  // v_sideways
  if (lhs->v_sideways != rhs->v_sideways) {
    return false;
  }
  // w_downward
  if (lhs->w_downward != rhs->w_downward) {
    return false;
  }
  // airspeed
  if (lhs->airspeed != rhs->airspeed) {
    return false;
  }
  // v_north
  if (lhs->v_north != rhs->v_north) {
    return false;
  }
  // v_east
  if (lhs->v_east != rhs->v_east) {
    return false;
  }
  // v_down
  if (lhs->v_down != rhs->v_down) {
    return false;
  }
  // climb_rate
  if (lhs->climb_rate != rhs->climb_rate) {
    return false;
  }
  // ground_speed
  if (lhs->ground_speed != rhs->ground_speed) {
    return false;
  }
  // p_roll_rate
  if (lhs->p_roll_rate != rhs->p_roll_rate) {
    return false;
  }
  // q_pitch_rate
  if (lhs->q_pitch_rate != rhs->q_pitch_rate) {
    return false;
  }
  // r_yaw_rate
  if (lhs->r_yaw_rate != rhs->r_yaw_rate) {
    return false;
  }
  // p_deg_s
  if (lhs->p_deg_s != rhs->p_deg_s) {
    return false;
  }
  // q_deg_s
  if (lhs->q_deg_s != rhs->q_deg_s) {
    return false;
  }
  // r_deg_s
  if (lhs->r_deg_s != rhs->r_deg_s) {
    return false;
  }
  // alpha
  if (lhs->alpha != rhs->alpha) {
    return false;
  }
  // beta
  if (lhs->beta != rhs->beta) {
    return false;
  }
  // alpha_deg
  if (lhs->alpha_deg != rhs->alpha_deg) {
    return false;
  }
  // beta_deg
  if (lhs->beta_deg != rhs->beta_deg) {
    return false;
  }
  return true;
}

bool
my_interfaces__msg__AircraftData__copy(
  const my_interfaces__msg__AircraftData * input,
  my_interfaces__msg__AircraftData * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // north
  output->north = input->north;
  // east
  output->east = input->east;
  // down
  output->down = input->down;
  // altitude
  output->altitude = input->altitude;
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // yaw
  output->yaw = input->yaw;
  // roll_deg
  output->roll_deg = input->roll_deg;
  // pitch_deg
  output->pitch_deg = input->pitch_deg;
  // yaw_deg
  output->yaw_deg = input->yaw_deg;
  // u_forward
  output->u_forward = input->u_forward;
  // v_sideways
  output->v_sideways = input->v_sideways;
  // w_downward
  output->w_downward = input->w_downward;
  // airspeed
  output->airspeed = input->airspeed;
  // v_north
  output->v_north = input->v_north;
  // v_east
  output->v_east = input->v_east;
  // v_down
  output->v_down = input->v_down;
  // climb_rate
  output->climb_rate = input->climb_rate;
  // ground_speed
  output->ground_speed = input->ground_speed;
  // p_roll_rate
  output->p_roll_rate = input->p_roll_rate;
  // q_pitch_rate
  output->q_pitch_rate = input->q_pitch_rate;
  // r_yaw_rate
  output->r_yaw_rate = input->r_yaw_rate;
  // p_deg_s
  output->p_deg_s = input->p_deg_s;
  // q_deg_s
  output->q_deg_s = input->q_deg_s;
  // r_deg_s
  output->r_deg_s = input->r_deg_s;
  // alpha
  output->alpha = input->alpha;
  // beta
  output->beta = input->beta;
  // alpha_deg
  output->alpha_deg = input->alpha_deg;
  // beta_deg
  output->beta_deg = input->beta_deg;
  return true;
}

my_interfaces__msg__AircraftData *
my_interfaces__msg__AircraftData__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__AircraftData * msg = (my_interfaces__msg__AircraftData *)allocator.allocate(sizeof(my_interfaces__msg__AircraftData), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(my_interfaces__msg__AircraftData));
  bool success = my_interfaces__msg__AircraftData__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
my_interfaces__msg__AircraftData__destroy(my_interfaces__msg__AircraftData * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    my_interfaces__msg__AircraftData__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
my_interfaces__msg__AircraftData__Sequence__init(my_interfaces__msg__AircraftData__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__AircraftData * data = NULL;

  if (size) {
    data = (my_interfaces__msg__AircraftData *)allocator.zero_allocate(size, sizeof(my_interfaces__msg__AircraftData), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = my_interfaces__msg__AircraftData__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        my_interfaces__msg__AircraftData__fini(&data[i - 1]);
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
my_interfaces__msg__AircraftData__Sequence__fini(my_interfaces__msg__AircraftData__Sequence * array)
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
      my_interfaces__msg__AircraftData__fini(&array->data[i]);
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

my_interfaces__msg__AircraftData__Sequence *
my_interfaces__msg__AircraftData__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__AircraftData__Sequence * array = (my_interfaces__msg__AircraftData__Sequence *)allocator.allocate(sizeof(my_interfaces__msg__AircraftData__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = my_interfaces__msg__AircraftData__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
my_interfaces__msg__AircraftData__Sequence__destroy(my_interfaces__msg__AircraftData__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    my_interfaces__msg__AircraftData__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
my_interfaces__msg__AircraftData__Sequence__are_equal(const my_interfaces__msg__AircraftData__Sequence * lhs, const my_interfaces__msg__AircraftData__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!my_interfaces__msg__AircraftData__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
my_interfaces__msg__AircraftData__Sequence__copy(
  const my_interfaces__msg__AircraftData__Sequence * input,
  my_interfaces__msg__AircraftData__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(my_interfaces__msg__AircraftData);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    my_interfaces__msg__AircraftData * data =
      (my_interfaces__msg__AircraftData *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!my_interfaces__msg__AircraftData__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          my_interfaces__msg__AircraftData__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!my_interfaces__msg__AircraftData__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
