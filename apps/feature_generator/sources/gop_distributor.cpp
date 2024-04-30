#include "gop_distributor.hpp"

struct gop_struct_node
{
  int disp_idx;
  int dec_idx;
  int ref0; // display_order
  int ref1; // display_order
};

static gop_struct_node g_pic_struct[3][3];

gop_distributor::gop_distributor(int keyint)
{
  this->keyint = keyint;
  gop_images.reserve(keyint);

  g_pic_struct[0][0] = {.disp_idx = 0, .dec_idx = 0, .ref0 = -1, .ref1 = -1};

  g_pic_struct[1][0] = {.disp_idx = 1, .dec_idx = 0, .ref0 = -1, .ref1 = -1};
  g_pic_struct[1][1] = {.disp_idx = 0, .dec_idx = 1, .ref0 = -1, .ref1 = 1};

  g_pic_struct[2][0] = {.disp_idx = 2, .dec_idx = 0, .ref0 = -1, .ref1 = -1};
  g_pic_struct[2][1] = {.disp_idx = 0, .dec_idx = 1, .ref0 = -1, .ref1 = 2};
  g_pic_struct[2][2] = {.disp_idx = 1, .dec_idx = 2, .ref0 = 0, .ref1 = 2};
}

void gop_distributor::put(std::shared_ptr<uint8_t> image)
{
  if (gop_poc && (gop_poc + 1) % keyint == 0)
  {
    gop_images[gop_poc++] = image;
    dispense_gop();
    gop_poc = 0;
    return;
  }
  else
    gop_images[gop_poc++] = image;
}

void gop_distributor::terminate()
{
}

bool gop_distributor::dequeue(gop_output &out)
{
  if (fifo_gop.empty())
  {
    return false;
  }

  out = fifo_gop.front();
  fifo_gop.pop();
  return true;
}

void gop_distributor::dispense_gop()
{
  gop_output gop{keyint};

  for (int i = 0; i < gop.num_pictures; ++i)
  {
    gop.pictures[i]->poc = (fifo_gop.size() * gop.num_pictures) + i;
    gop.pictures[i]->image = gop_images[i];
    if (i)
    {
      gop.pictures[i]->ref_image0 = i - 1 >= 0 ? gop_images[i - 1] : nullptr;
      gop.pictures[i]->ref_image1 = i + 1 < gop.num_pictures ? gop_images[i + 1] : nullptr;
    }
  }

  fifo_gop.push(gop);
}
