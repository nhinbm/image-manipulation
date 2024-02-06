#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <math.h>

using namespace std;
using namespace cv;

// 1. Load the input image from its path.
Mat load_image(char* input_file) {
  Mat image = imread(input_file);
  return image;
}

// 2. Save image
void save_image(Mat image, char* output_file) {
  namedWindow ("Show Image");
  imshow("Show Image", image);
  waitKey(0);
  imwrite(output_file, image);
}

// 3. Convert a color image to the gray image.
Mat rgb2gray(Mat input_image) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input, width_input, CV_8UC1);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  uchar* p_data_input = (uchar*)input_image.data;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      p_row_output[0] = 0.0722 * p_row_input[0] + 0.7152 * p_row_input[1] + 0.2126 * p_row_input[2];
    }
  }

  return output_image;
}

// 4. Change the brightness of an image.
Mat brightness(Mat input_image, float brightness_factor) {
  // Create constant hash table of 
  unordered_map<float, float> brightness_table;

  for (int i = 0; i <= 255; i++) {
    brightness_table[i] = min(max(i + i * brightness_factor, 0.0f), 255.0f);
  }

  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image;
  if (n_channels_input == 1) {
    output_image = Mat(height_input, width_input, CV_8UC1);
  } else if (n_channels_input == 3) {
    output_image = Mat(height_input, width_input, CV_8UC3);
  }
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  uchar* p_data_input = (uchar*)input_image.data;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = brightness_table[p_row_input[c]];
      }
    }
  }

  return output_image;
}

// 5. Change the contrast of an image.
Mat contrast(Mat input_image, float contrast_factor) {
  // Create constant hash table of 
  unordered_map<float, float> contrast_table;

  for (int i = 0; i <= 255; i++) {
    float factor = (259.0f * (255.0f * contrast_factor + 255.0f)) / (255.0f * (259.0f - 255.0f * contrast_factor));
    contrast_table[i] = min(max(factor * (i - 128) + 128, 0.0f), 255.0f);
  }

  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image;
  if (n_channels_input == 1) {
    output_image = Mat(height_input, width_input, CV_8UC1);
  } else if (n_channels_input == 3) {
    output_image = Mat(height_input, width_input, CV_8UC3);
  }
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  uchar* p_data_input = (uchar*)input_image.data;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = contrast_table[p_row_input[c]];
      }
    }
  }

  return output_image;
}

// 6. Filter an image using average filter.
Mat filter_avg(Mat input_image, int kernel_size) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image;
  if (n_channels_input == 1) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC1);
  } else if (n_channels_input == 3) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC3);
  }
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding = kernel_size / 2 * n_channels_input;
  uchar* p_data_input = (uchar*)input_image.data + padding + width_step_input * (kernel_size/2);
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      // Loop through kernel
      vector<float> avg_kernel(n_channels_input, 0.0);
      uchar* p_pixel = p_row_input - width_step_input * (kernel_size / 2) - padding;
      for (int ky = 0; ky < kernel_size; ky++, p_pixel += width_step_input) {
        uchar* p_row_pixel = p_pixel;
        for (int kx = 0; kx < kernel_size; kx++, p_row_pixel += n_channels_input) {
          for (int c = 0; c < n_channels_input; c++) {
            avg_kernel[c] += p_row_pixel[c] / (kernel_size * kernel_size);
          }
        }
      }
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = avg_kernel[c];
      }
    }
  }

  return output_image;
}

// 7. Filter an image using median filter.
Mat filter_med(Mat input_image, int kernel_size) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image;
  if (n_channels_input == 1) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC1);
  } else if (n_channels_input == 3) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC3);
  }
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding = kernel_size / 2 * n_channels_input; 
  uchar* p_data_input = (uchar*)input_image.data + padding + width_step_input * (kernel_size / 2);
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      // Loop through color
      for (int c = 0; c < n_channels_input; c++) {
        uchar* p_pixel = p_row_input - width_step_input * (kernel_size / 2) - padding;
        vector<float> med_kernel;
        // Loop through kernel
        for (int ky = 0; ky < kernel_size; ky++, p_pixel += width_step_input) {
          uchar* p_row_pixel = p_pixel;
          for (int kx = 0; kx < kernel_size; kx++, p_row_pixel += n_channels_input) {
            med_kernel.push_back(p_row_pixel[c]);
          }
        }
        auto middle = med_kernel.begin() + med_kernel.size() / 2;
        nth_element(med_kernel.begin(), middle, med_kernel.end());
        int median = *middle;
        p_row_output[c] = median;
      }
    }
  }

  return output_image;
}

// 8. Filter an image using gaussian filter.
Mat filter_gau(Mat input_image, int kernel_size) {
  // Create gaussian kernel
  float sigma = 1.0;
  vector<vector<double>> gau_kernel(kernel_size, vector<double>(kernel_size, 0.0));
  double sum = 0.0;
  for (int x = -kernel_size/2; x <= kernel_size/2; ++x) {
    for (int y = -kernel_size/2; y <= kernel_size/2; ++y) {
      double exponent = -(x * x + y * y) / (2 * sigma * sigma);
      gau_kernel[x + kernel_size/2][y + kernel_size/2] = (1 / (2 * 3.14 * sigma * sigma)) * exp(exponent);
      sum += gau_kernel[x + kernel_size/2][y + kernel_size/2];
    }
  }
  for (int i = 0; i < kernel_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      gau_kernel[i][j] /= sum;
    }
  }

  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image;
  if (n_channels_input == 1) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC1);
  } else if (n_channels_input == 3) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC3);
  }
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding = kernel_size / 2 * n_channels_input; 
  uchar* p_data_input = (uchar*)input_image.data + padding + width_step_input * (kernel_size / 2);
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      // Loop through kernel
      vector<float> gau_sum(n_channels_input, 0.0);
      uchar* p_pixel = p_row_input - width_step_input * (kernel_size / 2) - padding;
      for (int ky = 0; ky < kernel_size; ky++, p_pixel += width_step_input) {
        uchar* p_row_pixel = p_pixel;
        for (int kx = 0; kx < kernel_size; kx++, p_row_pixel += n_channels_input) {
          for (int c = 0; c < n_channels_input; c++) {
            gau_sum[c] += p_row_pixel[c] * gau_kernel[ky][kx];
          }
        }
      }
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = gau_sum[c];
      }
    }
  }

  return output_image;
}

// 9. Detect edge of an image using Sobel of kernel size 3 × 3.
Mat detect_sobel(Mat input_image) {
  // Create kernel
  int kernel_size = 3;
  vector<vector<int>> sobel_kernel_x = {{-1, 0, 1},
                                        {-2, 0, 2},
                                        {-1, 0, 1}};
  vector<vector<int>> sobel_kernel_y = {{-1, -2, -1},
                                        {0, 0, 0},
                                        {1, 2, 1}};

  // Grayscale input image
  input_image = rgb2gray(input_image);

  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC1);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding = kernel_size / 2 * n_channels_input; 
  uchar* p_data_input = (uchar*)input_image.data + padding + width_step_input * (kernel_size / 2);
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      // Loop through kernel
      vector<float> sum_kernel_x(n_channels_input, 0.0);
      vector<float> sum_kernel_y(n_channels_input, 0.0);
      uchar* p_pixel = p_row_input - width_step_input * (kernel_size / 2) - padding;
      for (int ky = 0; ky < kernel_size; ky++, p_pixel += width_step_input) {
        uchar* p_row_pixel = p_pixel;
        for (int kx = 0; kx < kernel_size; kx++, p_row_pixel += n_channels_input) {
          for (int c = 0; c < n_channels_input; c++) {
            sum_kernel_x[c] += p_row_pixel[c] * sobel_kernel_x[ky][kx];
            sum_kernel_y[c] += p_row_pixel[c] * sobel_kernel_y[ky][kx];
          }
        }
      }
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = sqrt(pow(sum_kernel_x[c], 2) + pow(sum_kernel_y[c], 2));
      }
    }
  }

  return output_image;
}

// 10. Detect edge of an image using Laplace of kernel size 3 × 3.
Mat detect_laplace(Mat input_image) {
  // Create kernel
  int kernel_size = 3;
  vector<vector<int>> laplace_kernel = {{-1, 0, 1},
                                        {-2, 0, 2},
                                        {-1, 0, 1}};

  // Grayscale input image
  input_image = rgb2gray(input_image);

  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC1);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding = kernel_size / 2 * n_channels_input; 
  uchar* p_data_input = (uchar*)input_image.data + padding + width_step_input * (kernel_size / 2);
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      // Loop through kernel
      vector<float> sum_laplace(n_channels_input, 0.0);
      uchar* p_pixel = p_row_input - width_step_input * (kernel_size / 2) - padding;
      for (int ky = 0; ky < kernel_size; ky++, p_pixel += width_step_input) {
        uchar* p_row_pixel = p_pixel;
        for (int kx = 0; kx < kernel_size; kx++, p_row_pixel += n_channels_input) {
          for (int c = 0; c < n_channels_input; c++) {
            sum_laplace[c] += p_row_pixel[c] * laplace_kernel[ky][kx];
          }
        }
      }
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = sum_laplace[c];
      }
    }
  }

  return output_image;
}

int main(int argc, char* argv[]) {
  // Information of command line arguments
  char* command = argv[1];
  char* input_file = argv[2];
  char* output_file = argv[3];

  // Load image
  Mat image = load_image(input_file);
  if (image.empty()) {
    cout << "Could not open the image!" << endl;
    return -1;
  }

  Mat output;

  if (strcmp(command, "-rgb2gray") == 0) {
    output = rgb2gray(image);
  } else if (strcmp(command, "-brightness") == 0) {
    float brightness_factor = stof(argv[4]);
    if (brightness_factor < -1 || brightness_factor > 1) {
      cout << "Input the brightness factor from 0 to 1" << endl;
      return -1;
    }
    output = brightness(image, brightness_factor);
  } else if (strcmp(command, "-contrast") == 0) {
    float contrast_factor = stof(argv[4]);
    if (contrast_factor < -1 || contrast_factor > 1) {
      cout << "Input the brightness factor from 0 to 1" << endl;
      return -1;
    }
    output = contrast(image, contrast_factor);
  } else if (strcmp(command, "-avg") == 0) {
    float kernel_size = stoi(argv[4]);
    output = filter_avg(image, kernel_size);
  } else if (strcmp(command, "-med") == 0) {
    float kernel_size = stoi(argv[4]);
    output = filter_med(image, kernel_size);
  } else if (strcmp(command, "-gau") == 0) {
    float kernel_size = stoi(argv[4]);
    output = filter_gau(image, kernel_size);
  } else if (strcmp(command, "-sobel") == 0) {
    output = detect_sobel(image);
  } else if (strcmp(command, "-laplace") == 0) {
    output = detect_laplace(image);
  } else {
    cout << "Unknown command\n";
  }

  save_image(output, output_file);

  return 0;
}