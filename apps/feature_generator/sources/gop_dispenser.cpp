#include "gop_dispenser.hpp"

struct gop_struct_node
{
  int disp_idx;
  int dec_idx;
  int ref0;   // display_order
  int ref1;   // display_order
};

static gop_struct_node g_pic_struct[4][4];

gop_dispenser::gop_dispenser(int bframes, int keyint)
{
  this->keyint = keyint;
  interval = bframes + 1;
  poc = 0;

  g_pic_struct[0][0] = {.disp_idx = 0, .dec_idx = 0, .ref0 = -1, .ref1 = -1};

  g_pic_struct[1][0] = {.disp_idx = 1, .dec_idx = 0, .ref0 = -1, .ref1 = -1};
  g_pic_struct[1][1] = {.disp_idx = 0, .dec_idx = 1, .ref0 = -1, .ref1 = 1};
  
  g_pic_struct[2][0] = {.disp_idx = 2, .dec_idx = 0, .ref0 = -1, .ref1 = -1};
  g_pic_struct[2][1] = {.disp_idx = 0, .dec_idx = 1, .ref0 = -1, .ref1 = 2};
  g_pic_struct[2][2] = {.disp_idx = 1, .dec_idx = 2, .ref0 = 0, .ref1 = 2};

  g_pic_struct[3][0] = {.disp_idx = 3, .dec_idx = 0, .ref0 = -1, .ref1 = -1};
  g_pic_struct[3][1] = {.disp_idx = 1, .dec_idx = 1, .ref0 = -1, .ref1 = 3};
  g_pic_struct[3][2] = {.disp_idx = 0, .dec_idx = 2, .ref0 = -1, .ref1 = 1};
  g_pic_struct[3][3] = {.disp_idx = 2, .dec_idx = 3, .ref0 = 1, .ref1 = 3};
}

void gop_dispenser::put(std::shared_ptr<uint8_t> image)
{
  if (poc % keyint == 0)
  {
    dispense_gop();
    gop_images[num_holding_pictures++] = image;
    leading_gop = true;
    gop_start_poc = poc;
  }
  else if (num_holding_pictures == (interval + (leading_gop ? 1 : 0)))
  {
    dispense_gop();
    gop_images[num_holding_pictures++] = image;
    leading_gop = false;
    gop_start_poc = poc;
  }
  else
  {
    gop_images[num_holding_pictures++] = image;
  }
  
  ++poc;
}

void gop_dispenser::terminate() 
{
  dispense_gop(); 
}

bool gop_dispenser::dequeue(gop_output& out)
{
  if(fifo_gop.empty())
  {
    return false;
  }

  out = fifo_gop.front();
  fifo_gop.pop();
  return true;
}

void gop_dispenser::dispense_gop()
{
  if (num_holding_pictures == 0)
  {
    return;
  }

  std::shared_ptr<uint8_t> prev_ref = last_image;
  gop_output gop = {};
  
  gop.num_pictures = num_holding_pictures;
  gop.is_leading = leading_gop;
  
  if (leading_gop)
  {
    gop.pictures[0].poc = gop_start_poc;
    gop.pictures[0].image = gop_images[0];
    gop.pictures[0].ref_image0 = nullptr;
    gop.pictures[0].ref_image1 = nullptr;
    prev_ref = gop_images[0];
    --num_holding_pictures;
  }

  const gop_struct_node* pGS = &g_pic_struct[num_holding_pictures - 1][0];
  int offset = leading_gop ? 1 : 0;

  for(int i = 0; i < num_holding_pictures; ++i)
  {
    int oidx = pGS[i].disp_idx + offset;
    
    gop.pictures[oidx].poc = gop_start_poc + oidx;
    gop.pictures[oidx].image = gop_images[oidx];
    gop.pictures[oidx].ref_image0 = (pGS[i].ref0 == -1) ? prev_ref : gop_images[pGS[i].ref0 + offset];
    gop.pictures[oidx].ref_image1 = nullptr;

    if(pGS[i].ref0 != -1 || pGS[i].ref1 != -1)
    {
      gop.pictures[oidx].ref_image1 = (pGS[i].ref1 == -1) ? prev_ref : gop_images[pGS[i].ref1 + offset];
    }
  }

  last_image = gop.pictures[gop.num_pictures - 1].image;
  fifo_gop.emplace(std::forward<gop_output>(gop));

  for(int i = 0 ; i < (int)(sizeof(gop_images)/ sizeof(decltype(gop_images[0]))); ++i)
  {
    gop_images[i] = nullptr;
  }
  num_holding_pictures = 0;
}