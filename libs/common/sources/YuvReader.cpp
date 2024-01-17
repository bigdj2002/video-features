#include "YuvReader.h"
#include <iostream>

bool YuvReader::open(const std::string& path, int width, int height, int bitdepth, int yuvformat)
{
  close();

  m_file.open(path, std::ios::binary);

  if(!m_file.is_open())
  {
    std::cerr << "YuvReader: \"" << path << "\" open failed" << std::endl;
    return false;
  }

  if(yuvformat != 420 && yuvformat != 422 && yuvformat != 444)
  {
    std::cerr << "YuvReader: unsupported yuv format " << yuvformat << std::endl;
    return false;
  }

  uint32_t left_shift = ((bitdepth + 7u) >> 3u) - 1u;
  uint32_t luma_frame_size = (width * height) << left_shift;
  uint32_t chroma_width = (yuvformat == 444) ? width : ((width + 1) >> 1);
  uint32_t chroma_height = (yuvformat == 420) ? ((height + 1) >> 1) : height;
  uint32_t chroma_frame_size = (chroma_width * chroma_height) << left_shift;

  uint32_t frame_size = luma_frame_size + chroma_frame_size * 2u;
  m_buffer.resize(frame_size);

  return true;
}

const uint8_t* YuvReader::read_next()
{
  if(!m_file.is_open())
  {
    return nullptr;
  }

  m_file.read((char*)m_buffer.data(), m_buffer.size());

  if((std::size_t)m_file.gcount() < m_buffer.size())
  {
    m_file.close();
    return nullptr;
  }

  return m_buffer.data();
}

void YuvReader::close()
{
  m_file.close();
  m_buffer.clear();
}