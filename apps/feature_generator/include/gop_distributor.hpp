#pragma once

#include <memory>
#include <cstdint>
#include <queue>
#include <vector>

struct picture_node
{
  int poc;
  std::shared_ptr<uint8_t> image = {};
  std::shared_ptr<uint8_t> ref_image0 = {};
  std::shared_ptr<uint8_t> ref_image1 = {};
};

struct gop_output
{
  picture_node pictures[16];
  int num_pictures;
  bool is_leading;
};

class gop_distributor
{
public:
  gop_distributor(int bframes, int keyint);

  // put images in display order
  void put(std::shared_ptr<uint8_t> image);

  void terminate();

  bool dequeue(gop_output& out);

private:
  int poc = 0;
  int interval = 0;
  int keyint = 0;
  int num_holding_pictures = 0;
  int gop_start_poc = 0;
  std::shared_ptr<uint8_t> gop_images[32];
  std::shared_ptr<uint8_t> last_image = {};
  bool leading_gop = false;
  std::queue<gop_output> fifo_gop;

private:
  void dispense_gop();
};