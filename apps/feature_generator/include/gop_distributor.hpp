#pragma once

#include <memory>
#include <cstdint>
#include <queue>
#include <vector>

struct picture_node
{
  int poc;
  std::shared_ptr<uint8_t> image;
  std::shared_ptr<uint8_t> ref_image0;
  std::shared_ptr<uint8_t> ref_image1;
};

struct gop_output
{
  std::vector<std::shared_ptr<picture_node>> pictures;
  int num_pictures;

public:
  gop_output(int keyint)
  {
    pictures.resize(keyint);
    for (int i = 0; i < keyint; ++i)
    {
      pictures[i] = std::make_shared<picture_node>();
    }
    num_pictures = keyint;
  }
};

class gop_distributor
{
public:
  gop_distributor(int keyint);

  // put images in display order
  void put(std::shared_ptr<uint8_t> image);

  void terminate();

  bool dequeue(gop_output& out);

private:
  int gop_poc = 0;
  int interval = 0;
  int keyint = 0;
  std::vector<std::shared_ptr<uint8_t>> gop_images;
  std::queue<gop_output> fifo_gop;

private:
  void dispense_gop();
};