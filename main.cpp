#ifdef _WIN32
#include <Windows.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <GL/glew.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdio.h>
#include <vector>
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

struct FRAME
{
  FRAME() :
    framebuffer_(GL_INVALID_VALUE),
    texture_(GL_INVALID_VALUE)
  {
  }

  GLuint framebuffer_;
  GLuint texture_;
};

void GenerateLinearLUT(const int video_width, const int video_height, uint8_t* dewarp_lut)
{
  for (int y = 0; y < video_height; ++y)
  {
    for (int x = 0; x < video_width; ++x)
    {
      const uint16_t value_x = static_cast<uint16_t>((static_cast<float>(x) / static_cast<float>(video_width)) * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const uint16_t value_y = static_cast<uint16_t>((static_cast<float>(y) / static_cast<float>(video_height)) * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const int index = ((y * video_width) + x) * 4;
      dewarp_lut[index] = reinterpret_cast<const uint8_t*>(&value_x)[0];
      dewarp_lut[index + 1] = reinterpret_cast<const uint8_t*>(&value_x)[1];
      dewarp_lut[index + 2] = reinterpret_cast<const uint8_t*>(&value_y)[0];
      dewarp_lut[index + 3] = reinterpret_cast<const uint8_t*>(&value_y)[1];
    }
  }
}

void GenerateUndistortLUT(const float zoom, const int video_width, const int video_height, const cv::Mat& camera_matrix, const cv::Mat& distortion_coeffs, uint8_t* dewarp_lut)
{
  cv::Mat map1;
  cv::Mat map2;
  cv::initUndistortRectifyMap(camera_matrix, distortion_coeffs, cv::Mat(), camera_matrix, cv::Size(video_width, video_height), CV_32FC1, map1, map2);
  for (int y = 0; y < video_height; ++y)
  {
    for (int x = 0; x < video_width; ++x)
    {
      float undistorted_x = map1.at<float>(y, x) / static_cast<float>(video_width);
      float undistorted_y = map2.at<float>(y, x) / static_cast<float>(video_height);
      undistorted_x = std::max(std::min(((undistorted_x - 0.5f) * zoom) + 0.5f, 1.0f), 0.0f);
      undistorted_y = std::max(std::min(((undistorted_y - 0.5f) * zoom) + 0.5f, 1.0f), 0.0f);
      const uint16_t value_x = static_cast<uint16_t>(undistorted_x * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const uint16_t value_y = static_cast<uint16_t>(undistorted_y * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const int index = ((y * video_width) + x) * 4;
      dewarp_lut[index] = reinterpret_cast<const uint8_t*>(&value_x)[0];
      dewarp_lut[index + 1] = reinterpret_cast<const uint8_t*>(&value_x)[1];
      dewarp_lut[index + 2] = reinterpret_cast<const uint8_t*>(&value_y)[0];
      dewarp_lut[index + 3] = reinterpret_cast<const uint8_t*>(&value_y)[1];
    }
  }
}

void GenerateFisheyeLUT(const float zoom, const int video_width, const int video_height, const cv::Mat& camera_matrix, const cv::Mat& distortion_coeffs, uint8_t* dewarp_lut)
{
  cv::Mat map1;
  cv::Mat map2;
  cv::fisheye::initUndistortRectifyMap(camera_matrix, distortion_coeffs, cv::Mat(), camera_matrix, cv::Size(video_width, video_height), CV_32FC1, map1, map2);
  for (int y = 0; y < video_height; ++y)
  {
    for (int x = 0; x < video_width; ++x)
    {
      float undistorted_x = map1.at<float>(y, x) / static_cast<float>(video_width);
      float undistorted_y = map2.at<float>(y, x) / static_cast<float>(video_height);
      undistorted_x = std::max(std::min(((undistorted_x - 0.5f) * zoom) + 0.5f, 1.0f), 0.0f);
      undistorted_y = std::max(std::min(((undistorted_y - 0.5f) * zoom) + 0.5f, 1.0f), 0.0f);
      const uint16_t value_x = static_cast<uint16_t>(undistorted_x * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const uint16_t value_y = static_cast<uint16_t>(undistorted_y * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const int index = ((y * video_width) + x) * 4;
      dewarp_lut[index] = reinterpret_cast<const uint8_t*>(&value_x)[0];
      dewarp_lut[index + 1] = reinterpret_cast<const uint8_t*>(&value_x)[1];
      dewarp_lut[index + 2] = reinterpret_cast<const uint8_t*>(&value_y)[0];
      dewarp_lut[index + 3] = reinterpret_cast<const uint8_t*>(&value_y)[1];
    }
  }
}

void GenerateOmnidirectionalLUT(const float zoom, const float xi, const int video_width, const int video_height, const cv::Mat& camera_matrix, const cv::Mat& distortion_coeffs, uint8_t* dewarp_lut)
{
  cv::Mat map1;
  cv::Mat map2;
  cv::omnidir::initUndistortRectifyMap(camera_matrix, distortion_coeffs, xi, cv::Matx33d::eye(), camera_matrix, cv::Size(video_width, video_height), CV_32FC1, map1, map2, cv::omnidir::RECTIFY_PERSPECTIVE);
  for (int y = 0; y < video_height; ++y)
  {
    for (int x = 0; x < video_width; ++x)
    {
      float undistorted_x = map1.at<float>(y, x) / static_cast<float>(video_width);
      float undistorted_y = map2.at<float>(y, x) / static_cast<float>(video_height);
      undistorted_x = std::max(std::min(((undistorted_x - 0.5f) * zoom) + 0.5f, 1.0f), 0.0f);
      undistorted_y = std::max(std::min(((undistorted_y - 0.5f) * zoom) + 0.5f, 1.0f), 0.0f);
      const uint16_t value_x = static_cast<uint16_t>(undistorted_x * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const uint16_t value_y = static_cast<uint16_t>(undistorted_y * static_cast<float>(std::numeric_limits<uint16_t>::max()));
      const int index = ((y * video_width) + x) * 4;
      dewarp_lut[index] = reinterpret_cast<const uint8_t*>(&value_x)[0];
      dewarp_lut[index + 1] = reinterpret_cast<const uint8_t*>(&value_x)[1];
      dewarp_lut[index + 2] = reinterpret_cast<const uint8_t*>(&value_y)[0];
      dewarp_lut[index + 3] = reinterpret_cast<const uint8_t*>(&value_y)[1];
    }
  }
}

int main(int argc, char** argv)
{
  // Check command line arguments
  if (argc != 2)
  {
    std::cerr << "Usage:\nImGuiPlayer video.mp4" << std::endl;
    return -1;
  }
  // Parse file with FFMPEG
  AVFormatContext* format_context = nullptr;
  AVDictionary* options = nullptr;
  av_dict_set(&options, "rtsp_transport", "tcp", 0);
  av_dict_set(&options, "stimeout", "5000000", 0);
  av_dict_set(&options, "analyzeduration", "10000000", 0);
  av_dict_set(&options, "probesize", "50M", 0);
  if (avformat_open_input(&format_context, argv[1], nullptr, &options) < 0)
  {
    std::cerr << "Failed to open file" << std::endl;
    return -1;
  }
  av_dict_free(&options);
  if (avformat_find_stream_info(format_context, nullptr) < 0)
  {
    std::cerr << "Failed to find stream info" << std::endl;
    return -1;
  }
  std::optional<unsigned int> video_stream;
  for (unsigned int i = 0; i < format_context->nb_streams; i++)
  {
    if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
    {
      video_stream = i;
      break;
    }
  }
  if (!video_stream.has_value())
  {
    std::cerr << "Failed to find video stream" << std::endl;
    return -1;
  }
  // Open the decoder
  const AVCodec* codec = avcodec_find_decoder(format_context->streams[*video_stream]->codecpar->codec_id);
  if (codec == nullptr)
  {
    std::cerr << "Failed to find decoder" << std::endl;
    return -1;
  }
  AVCodecContext* codec_context = avcodec_alloc_context3(codec);
  if (codec_context == nullptr)
  {
    std::cerr << "Failed to allocate codec context" << std::endl;
    return -1;
  }
  // Copy codec parameters
  if (avcodec_parameters_to_context(codec_context, format_context->streams[*video_stream]->codecpar) < 0)
  {
    std::cerr << "Failed to copy codec parameters" << std::endl;
    return -1;
  }
  if (avcodec_open2(codec_context, codec, nullptr) < 0)
  {
    std::cerr << "Failed to open codec" << std::endl;
    return -1;
  }
  // Init window
  if (!glfwInit())
  {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  const int video_width = format_context->streams[*video_stream]->codecpar->width;
  const int video_height = format_context->streams[*video_stream]->codecpar->height;
  GLFWwindow* window = glfwCreateWindow(1600, 600, "Dewarping Player", nullptr, nullptr);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK)
  {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }
  glfwSwapInterval(1);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  int display_width = 0;
  int display_height = 0;
  glfwGetFramebufferSize(window, &display_width, &display_height);
  glViewport(0, 0, display_width, display_height);
  // Setup ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr; // Stop saving files
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
  // YUV shader
  const char* yuv_vertex_shader_source = R"(#version 330 core
                                            layout(location = 0) in vec2 in_pos;
                                            layout(location = 1) in vec2 in_tex_coord;
                                            out vec2 tex_coord;
                                            void main()
                                            {
                                                gl_Position = vec4(in_pos, 0.0, 1.0);
                                                tex_coord = in_tex_coord;
                                            })";
  const char* yuv_fragment_shader_source = R"(#version 330 core
                                              out vec4 FragColor;
                                              in vec2 tex_coord;
                                              uniform sampler2D texture_y;
                                              uniform sampler2D texture_u;
                                              uniform sampler2D texture_v;
                                              void main()
                                              {
                                                  float y = texture(texture_y, tex_coord).r;
                                                  float u = texture(texture_u, tex_coord).r - 0.5;
                                                  float v = texture(texture_v, tex_coord).r - 0.5;
                                                  vec3 rgb = mat3(1.0, 1.0, 1.0,
                                                                  0.0, -0.39465, 2.03211,
                                                                  1.13983, -0.58060, 0.0) * vec3(y, u, v);
                                                  FragColor = vec4(rgb, 1.0);
                                              })";
  const GLuint yuv_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(yuv_vertex_shader, 1, &yuv_vertex_shader_source, nullptr);
  glCompileShader(yuv_vertex_shader);
  const GLuint yuv_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(yuv_fragment_shader, 1, &yuv_fragment_shader_source, nullptr);
  glCompileShader(yuv_fragment_shader);
  const GLuint yuv_shader_program = glCreateProgram();
  glAttachShader(yuv_shader_program, yuv_vertex_shader);
  glAttachShader(yuv_shader_program, yuv_fragment_shader);
  glLinkProgram(yuv_shader_program);
  glDeleteShader(yuv_vertex_shader);
  glDeleteShader(yuv_fragment_shader);
  // YUV textures
  std::array<GLuint, 3> yuv_textures;
  glGenTextures(3, yuv_textures.data());
  for (int i = 0; i < 3; i++)
  {
    glBindTexture(GL_TEXTURE_2D, yuv_textures[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
  glBindTexture(GL_TEXTURE_2D, yuv_textures[0]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, video_width, video_height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, yuv_textures[1]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, video_width / 2, video_height / 2, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, yuv_textures[2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, video_width / 2, video_height / 2, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  // Dewarp shader
  const char* dewarp_vertex_shader_source = R"(#version 330 core
                                               layout(location = 0) in vec2 in_pos;
                                               layout(location = 1) in vec2 in_tex_coord;
                                               out vec2 tex_coord;
                                               void main()
                                               {
                                                 tex_coord = in_tex_coord;
                                                 gl_Position = vec4(in_pos, 0, 1);
                                               })";
  const char* dewarp_fragment_shader_source = R"(#version 330 core
                                                 precision mediump float;
                                                 in vec2 tex_coord;
                                                 out vec4 FragColor;
                                                 uniform sampler2D tex;
                                                 uniform sampler2D lut;
                                                 void main()
                                                 {
                                                   vec4 lut_value = texture(lut, tex_coord);
                                                   float x = lut_value.g + (lut_value.r / 255.0);
                                                   float y = lut_value.a + (lut_value.b / 255.0);
                                                   FragColor = texture(tex, vec2(x, y));
                                                 })";
  const GLuint dewarp_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(dewarp_vertex_shader, 1, &dewarp_vertex_shader_source, nullptr);
  glCompileShader(dewarp_vertex_shader);
  const GLuint dewarp_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(dewarp_fragment_shader, 1, &dewarp_fragment_shader_source, nullptr);
  glCompileShader(dewarp_fragment_shader);
  const GLuint dewarp_shader_program = glCreateProgram();
  glAttachShader(dewarp_shader_program, dewarp_vertex_shader);
  glAttachShader(dewarp_shader_program, dewarp_fragment_shader);
  glLinkProgram(dewarp_shader_program);
  glDeleteShader(dewarp_vertex_shader);
  glDeleteShader(dewarp_fragment_shader);
  // Dewarp textures
  std::unique_ptr<uint8_t[]> dewarp_lut = std::make_unique<uint8_t[]>(video_width * video_height * sizeof(uint16_t) * 2);
  GenerateLinearLUT(video_width, video_height, dewarp_lut.get());
  GLuint dewarp_lut_texture = GL_INVALID_VALUE;
  glGenTextures(1, &dewarp_lut_texture);
  glBindTexture(GL_TEXTURE_2D, dewarp_lut_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, video_width, video_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dewarp_lut.get());
  // YUV Geometory
  const float yuv_vertices[] =
  {
    // positions  texture coords
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
     1.0f,  1.0f, 1.0f, 1.0f,
    -1.0f,  1.0f, 0.0f, 1.0f
  };
  const unsigned int yuv_indices[] =
  {
    0, 1, 2,
    2, 3, 0
  };
  GLuint yuv_vao = GL_INVALID_VALUE;
  GLuint yuv_vbo = GL_INVALID_VALUE;
  GLuint yuv_ebo = GL_INVALID_VALUE;
  glGenVertexArrays(1, &yuv_vao);
  glGenBuffers(1, &yuv_vbo);
  glGenBuffers(1, &yuv_ebo);
  glBindVertexArray(yuv_vao);
  glBindBuffer(GL_ARRAY_BUFFER, yuv_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(yuv_vertices), yuv_vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, yuv_ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(yuv_indices), yuv_indices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);
  // Dewarp geometory
  const float dewarp_vertices[] =
  {
    // positions  texture coords
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
     1.0f,  1.0f, 1.0f, 1.0f,
    -1.0f,  1.0f, 0.0f, 1.0f
  };
  const unsigned int dewarp_indices[] =
  {
    0, 1, 2,
    2, 3, 0
  };
  GLuint dewarp_vao = GL_INVALID_VALUE;
  GLuint dewarp_vbo = GL_INVALID_VALUE;
  GLuint dewarp_ebo = GL_INVALID_VALUE;
  glGenVertexArrays(1, &dewarp_vao);
  glGenBuffers(1, &dewarp_vbo);
  glGenBuffers(1, &dewarp_ebo);
  glBindVertexArray(dewarp_vao);
  glBindBuffer(GL_ARRAY_BUFFER, dewarp_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(dewarp_vertices), dewarp_vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, dewarp_ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(dewarp_indices), dewarp_indices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);
  // Frame buffers
  glViewport(0, 0, video_width, video_height);
  std::array<FRAME, 2> frames;
  for (FRAME& frame : frames)
  {
    glGenFramebuffers(1, &frame.framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, frame.framebuffer_);
    glGenTextures(1, &frame.texture_);
    glBindTexture(GL_TEXTURE_2D, frame.texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, video_width, video_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame.texture_, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
      std::cerr << "Failed to create frame buffer" << std::endl;
      return -1;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
  // Main loop
  AVPacket* av_packet = av_packet_alloc();
  AVFrame* av_frame = av_frame_alloc();
  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  while (!glfwWindowShouldClose(window))
  {
    // Decode and build some frames if needed
    int ret = av_read_frame(format_context, av_packet);
    if (ret && (ret != AVERROR_EOF))
    {
      break;
    }
    else
    {
      if (av_packet->stream_index != *video_stream)
      {
        continue;
      }
      // Send packets
      if (avcodec_send_packet(codec_context, av_packet))
      {
        break;
      }
    }
    // Collect frames
    glViewport(0, 0, video_width, video_height);
    while (true)
    {
      // Draw YUV
      if (avcodec_receive_frame(codec_context, av_frame))
      {
        break;
      }
      // Update textures with AVFrame data
      glBindFramebuffer(GL_FRAMEBUFFER, frames[0].framebuffer_);
      glUseProgram(yuv_shader_program);
      // Bind textures
      glBindTexture(GL_TEXTURE_2D, yuv_textures[0]);
      glPixelStorei(GL_UNPACK_ROW_LENGTH, av_frame->linesize[0]);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, av_frame->width, av_frame->height, GL_RED, GL_UNSIGNED_BYTE, av_frame->data[0]);
      glBindTexture(GL_TEXTURE_2D, yuv_textures[1]);
      glPixelStorei(GL_UNPACK_ROW_LENGTH, av_frame->linesize[1]);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, av_frame->width / 2, av_frame->height / 2, GL_RED, GL_UNSIGNED_BYTE, av_frame->data[1]);
      glBindTexture(GL_TEXTURE_2D, yuv_textures[2]);
      glPixelStorei(GL_UNPACK_ROW_LENGTH, av_frame->linesize[2]);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, av_frame->width / 2, av_frame->height / 2, GL_RED, GL_UNSIGNED_BYTE, av_frame->data[2]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, yuv_textures[0]);
      glUniform1i(glGetUniformLocation(yuv_shader_program, "texture_y"), 0);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, yuv_textures[1]);
      glUniform1i(glGetUniformLocation(yuv_shader_program, "texture_u"), 1);
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, yuv_textures[2]);
      glUniform1i(glGetUniformLocation(yuv_shader_program, "texture_v"), 2);
      // Draw
      glBindVertexArray(yuv_vao);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
      // Clean up
      glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
      glUseProgram(0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      // Draw dewarp
      glBindFramebuffer(GL_FRAMEBUFFER, frames[1].framebuffer_);
      glUseProgram(dewarp_shader_program);
      // Bind textures
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, frames[0].texture_);
      glUniform1i(glGetUniformLocation(dewarp_shader_program, "tex"), 0);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, dewarp_lut_texture);
      glUniform1i(glGetUniformLocation(dewarp_shader_program, "lut"), 1);
      // Draw
      glBindVertexArray(dewarp_vao);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
      // Clean up
      glUseProgram(0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    glViewport(0, 0, display_width, display_height);
    // ImGui stuff
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    // Window
    ImGui::NewFrame();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y));
    ImGui::Begin("Frame", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar);
    ImGui::Image(static_cast<ImTextureID>(frames[0].texture_), ImVec2(800, 600));
    ImGui::SameLine();
    ImGui::Image(static_cast<ImTextureID>(frames[1].texture_), ImVec2(800, 600));
    ImGui::End();
    ImGui::PopStyleVar(4);
    // Draw setup window
    ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Once);
    ImGui::Begin("Setup");
    static int current_mode = 0;
    const char* items[] = { "linear", "opencv undistort", "opencv fisheye", "opencv omnidir" };
    bool redraw = false;
    if (ImGui::BeginCombo("Mode", items[current_mode]))
    {
      for (int n = 0; n < IM_ARRAYSIZE(items); n++)
      {
        const bool is_selected = (current_mode == n);
        if (ImGui::Selectable(items[n], is_selected))
        {
          current_mode = n;
          redraw = true;
        }
      }
      ImGui::EndCombo();
    }
    if (current_mode == 0)
    {
      if (redraw)
      {
        GenerateLinearLUT(video_width, video_height, dewarp_lut.get());
        glBindTexture(GL_TEXTURE_2D, dewarp_lut_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, video_width, video_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dewarp_lut.get());
      }
    }
    else if (current_mode == 1)
    {
      static float zoom = 1.0f;
      if (ImGui::DragFloat("distort_zoom", &zoom, 0.01, 0.5f, 1.5f))
      {
        redraw = true;
      }
      static float focal_length = 1700.0f;
      if (ImGui::DragFloat("distort_focal_length", &focal_length, 1.0f, 500.0f, 3000.0f))
      {
        redraw = true;
      }
      static float tangential_1 = 0.0f;
      if (ImGui::DragFloat("distort_tangential_1", &tangential_1, 0.00001f, -0.01f, 0.01f, "%.4f"))
      {
        redraw = true;
      }
      static float tangential_2 = 0.0f;
      if (ImGui::DragFloat("distort_tangential_2", &tangential_2, 0.00001f, -0.01f, 0.01f, "%.4f"))
      {
        redraw = true;
      }
      static float radial_1 = -0.2f;
      if (ImGui::DragFloat("distort_radial_1", &radial_1, 0.001f, -1.0f, 1.0f))
      {
        redraw = true;
      }
      static float radial_2 = 0.04f;
      if (ImGui::DragFloat("distort_radial_2", &radial_2, 0.001f, -0.5f, 0.5f))
      {
        redraw = true;
      }
      static float radial_3 = 0.00f;
      if (ImGui::DragFloat("distort_radial_3", &radial_3, 0.001f, -0.5f, 0.5f))
      {
        redraw = true;
      }
      if (redraw)
      {
        const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0,            static_cast<float>(video_width) / 2.0f,
                                                                 0,            focal_length, static_cast<float>(video_height) / 2.0f,
                                                                 0,            0,            1);
        const cv::Mat distortion_coeffs = (cv::Mat_<double>(1, 5) << radial_1, radial_2, tangential_1, tangential_2, radial_3);
        GenerateUndistortLUT(zoom, video_width, video_height, camera_matrix, distortion_coeffs, dewarp_lut.get());
        glBindTexture(GL_TEXTURE_2D, dewarp_lut_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, video_width, video_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dewarp_lut.get());
      }
    }
    else if (current_mode == 2)
    {
      static float zoom = 1.0f;
      if (ImGui::DragFloat("fisheye_zoom", &zoom, 0.01, 0.5f, 1.5f))
      {
        redraw = true;
      }
      static float focal_length = 1700.0f;
      if (ImGui::DragFloat("fisheye_focal_length", &focal_length, 1.0f, 500.0f, 3000.0f))
      {
        redraw = true;
      }
      static float k1 = 0.0f;
      if (ImGui::DragFloat("fisheye_k1", &k1, 0.001f, -1.0f, 1.0f))
      {
        redraw = true;
      }
      static float k2 = 0.0f;
      if (ImGui::DragFloat("fisheye_k2", &k2, 0.001f, -1.0f, 1.0f))
      {
        redraw = true;
      }
      static float k3 = 0.0f;
      if (ImGui::DragFloat("fisheye_k3", &k3, 0.001f, -1.0f, 1.0f))
      {
        redraw = true;
      }
      static float k4 = 0.0f;
      if (ImGui::DragFloat("fisheye_k4", &k4, 0.001f, -1.0f, 1.0f))
      {
        redraw = true;
      }
      if (redraw)
      {
        const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0,            static_cast<float>(video_width) / 2.0f,
                                                                 0,            focal_length, static_cast<float>(video_height) / 2.0f,
                                                                 0,            0,            1);
        const cv::Mat distortion_coeffs = (cv::Mat_<double>(1, 4) << k1, k2, k3, k4);
        GenerateFisheyeLUT(zoom, video_width, video_height, camera_matrix, distortion_coeffs, dewarp_lut.get());
        glBindTexture(GL_TEXTURE_2D, dewarp_lut_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, video_width, video_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dewarp_lut.get());
      }
    }
    else if (current_mode == 3)
    {
      static float zoom = 1.0f;
      if (ImGui::DragFloat("omnidirectional_zoom", &zoom, 0.01, 0.5f, 4.0f))
      {
        redraw = true;
      }
      static float xi = 1.2f;
      if (ImGui::DragFloat("omnidirectional_xi", &xi, 0.01, 0.5f, 1.5f))
      {
        redraw = true;
      }
      static float focal_length = 1700.0f;
      if (ImGui::DragFloat("omnidirectional_focal_length", &focal_length, 1.0f, 500.0f, 3000.0f))
      {
        redraw = true;
      }
      static float k1 = 0.0f;
      if (ImGui::DragFloat("omnidirectional_k1", &k1, 0.001f, -4.5f, 4.5f))
      {
        redraw = true;
      }
      static float k2 = 0.0f;
      if (ImGui::DragFloat("omnidirectional_k2", &k2, 0.001f, -4.5f, 4.5f))
      {
        redraw = true;
      }
      static float p1 = 0.0f;
      if (ImGui::DragFloat("omnidirectional_p1", &p1, 0.001f, -0.5f, 0.5f))
      {
        redraw = true;
      }
      static float p2 = 0.0f;
      if (ImGui::DragFloat("omnidirectional_p2", &p2, 0.0001f, -0.05f, 0.05f))
      {
        redraw = true;
      }
      if (redraw)
      {
        const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0,            static_cast<float>(video_width) / 2.0f,
                                                                 0,            focal_length, static_cast<float>(video_height) / 2.0f,
                                                                 0,            0,            1);
        const cv::Mat distortion_coeffs = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);
        GenerateOmnidirectionalLUT(zoom, xi, video_width, video_height, camera_matrix, distortion_coeffs, dewarp_lut.get());
        glBindTexture(GL_TEXTURE_2D, dewarp_lut_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, video_width, video_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, dewarp_lut.get());
      }
    }
    ImGui::End();
    ImGui::EndFrame();
    // Rendering
    ImGui::Render();
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }
  // Cleanup
  glDeleteProgram(yuv_shader_program);
  glDeleteTextures(yuv_textures.size(), yuv_textures.data());
  glDeleteVertexArrays(1, &yuv_vao);
  glDeleteBuffers(1, &yuv_vbo);
  glDeleteBuffers(1, &yuv_ebo);
  glDeleteProgram(dewarp_shader_program);
  glDeleteTextures(1, &dewarp_lut_texture);
  glDeleteVertexArrays(1, &dewarp_vao);
  glDeleteBuffers(1, &dewarp_vbo);
  glDeleteBuffers(1, &dewarp_ebo);
  for (FRAME& frame : frames)
  {
    glDeleteFramebuffers(1, &frame.framebuffer_);
    glDeleteTextures(1, &frame.texture_);
  }
  // Codec
  avformat_free_context(format_context);
  avcodec_free_context(&codec_context);
  av_packet_free(&av_packet);
  av_frame_free(&av_frame);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}