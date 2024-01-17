#include <iostream>

#include "yuv_reader.h"

bool yuv_reader::open(const std::string& path, int width, int height, int bitdepth, int yuvformat)
{
  close();

  m_file.open(path, std::ios::binary);

  if(!m_file.is_open())
  {
    std::cerr << "yuv_reader: \"" << path << "\" open failed" << std::endl;
    return false;
  }

  if(yuvformat != 420 && yuvformat != 422 && yuvformat != 444)
  {
    std::cerr << "yuv_reader: unsupported yuv format " << yuvformat << std::endl;
    return false;
  }

  uint32_t left_shift = ((bitdepth + 7u) >> 3u) - 1u;
  uint32_t luma_frame_size = (width * height) << left_shift;
  uint32_t chroma_width = (yuvformat == 444) ? width : ((width + 1) >> 1);
  uint32_t chroma_height = (yuvformat == 420) ? ((height + 1) >> 1) : height;
  uint32_t chroma_frame_size = (chroma_width * chroma_height) << left_shift;

  uint32_t frame_size = luma_frame_size + chroma_frame_size * 2u;
  m_frame_size_in_bytes = frame_size;

  return true;
}

std::shared_ptr<uint8_t> yuv_reader::read_next()
{
  if(!m_file.is_open())
  {
    return nullptr;
  }

  std::shared_ptr<uint8_t> image;
  image.reset(new uint8_t[m_frame_size_in_bytes]);
  m_file.read((char*)image.get(), m_frame_size_in_bytes);

  if((std::size_t)m_file.gcount() < m_frame_size_in_bytes)
  {
    m_file.close();
    return nullptr;
  }

  return image;
}

void yuv_reader::close()
{
  m_file.close();
}