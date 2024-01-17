#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <fstream>

class yuv_reader
{
public:
  yuv_reader() = default;
  ~yuv_reader() = default;
  
  bool open(const std::string& path, int width, int height, int bitdepth = 8, int yuvformat=420);

public:
  std::shared_ptr<uint8_t> read_next();
  const uint32_t get_buffer_size() const { return m_frame_size_in_bytes; }
  void close();

private:
  std::ifstream m_file;
  uint32_t m_frame_size_in_bytes = 0u;
};