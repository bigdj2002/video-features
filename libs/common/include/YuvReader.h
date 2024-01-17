#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

class YuvReader
{
public:
  bool open(const std::string& path, int width, int height, int bitdepth, int yuvformat=420);

  const uint8_t* read_next();
  
  void close();

private:
  std::ifstream m_file;
  std::vector<uint8_t> m_buffer;
};