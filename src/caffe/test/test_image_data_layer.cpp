#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  
  virtual void SetUp() {
    this->dim_label_ = 4
    this->blob_top_vec_.push_back(this->blob_top_data_);
    this->blob_top_vec_.push_back(this->blob_top_label_);
    Caffe::set_random_seed(this->seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(this->filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << this->filename_;
    for (int i = 0; i < 5; ++i) {

      //outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << i << std::endl;
      // multi-label
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg"
      for (int label_id = 0; label_id < this->dim_label_; ++label_id) {
        outfile << " " << i * this->dim_label_ + label_id;
      }
      outfile << std::endl;

    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(this->filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << this->filename_reshape_;

    // reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0 << std::endl;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg"
    for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
      reshapefile << " " << 0 * this->dim_label_ + label_id;
    }
    reshapefile << std::endl;

    // reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 1 << std::endl;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg"
    for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
      reshapefile << " " << 1 * this->dim_label_ + label_id;
    }
    reshapefile << std::endl;

    reshapefile.close();
    // Create test input file for images with space in names
    MakeTempFilename(&filename_space_);
    std::ofstream spacefile(this->filename_space_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << this->filename_space_;

    // spacefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0 << std::endl;
    spacefile << EXAMPLES_SOURCE_DIR "images/cat.jpg"
    for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
      spacefile << " " << 0 * this->dim_label_ + label_id;
    }
    spacefile << std::endl;
    
    //spacefile << EXAMPLES_SOURCE_DIR "images/cat_gray.jpg " << 1 << std::endl;
    spacefile << EXAMPLES_SOURCE_DIR "images/cat_gray.jpg"
    for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
      spacefile << " " << 1 * this->dim_label_ + label_id;
    }
    spacefile << std::endl;

    spacefile.close();
  }

  virtual ~ImageDataLayerTest() {
    delete this->blob_top_data_;
    delete this->blob_top_label_;
  }

  int seed_;
  
  // multi-label
  int dim_label_;

  string filename_;
  string filename_reshape_;
  string filename_space_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);

  // multi-label
  image_data_param->set_dim_label(this->dim_label_);

  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);

  // EXPECT_EQ(this->blob_top_label_->channels(), 1);
  // multi-label
  EXPECT_EQ(this->blob_top_label_->channels(), this->dim_label_);

  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {

      //EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
        EXPECT_EQ(i + label_id, this->blob_top_label_->cpu_data()[i * this->dim_label_ + label_id]);
      }
    
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);

  // multi-label
  image_data_param->set_dim_label(this->dim_label_);

  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  
  // EXPECT_EQ(this->blob_top_label_->channels(), 1);
  // multi-label
  EXPECT_EQ(this->blob_top_label_->channels(), this->dim_label_);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {

      // EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
        EXPECT_EQ(i + label_id, this->blob_top_label_->cpu_data()[i * this->dim_label_ + label_id]);
      }
    
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);

  // multi-label
  image_data_param->set_dim_label(this->dim_label_);

  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  
  // EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), this->dim_label_);

  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);

  // multi-label
  image_data_param->set_dim_label(this->dim_label_);

  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  
  // EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), this->dim_label_);

  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {

      // Dtype value = this->blob_top_label_->cpu_data()[i];
      // // Check that the value has not been seen already (no duplicates).
      // EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      // values_to_indices[value] = i;
      // num_in_order += (value == Dtype(i));
      
      // multi-label
      for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
        Dtype value = this->blob_top_label_->cpu_data()[i * this->dim_label_ + label_id];  
        EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
        values_to_indices[value] = i * this->dim_label_ + label_id;
        num_in_order += (value == Dtype(i * this->dim_label_ + label_id));
      }

    }
    EXPECT_EQ(5 * this->dim_label_, values_to_indices.size());
    EXPECT_GT(5 * this->dim_label_, num_in_order);
  }
}

TYPED_TEST(ImageDataLayerTest, TestSpace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_space_.c_str());
  image_data_param->set_shuffle(false);

   // multi-label
  image_data_param->set_dim_label(this->dim_label_);

  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  
  // EXPECT_EQ(this->blob_top_label_->channels(), 1);
  // multi-label
  EXPECT_EQ(this->blob_top_label_->channels(), this->dim_label_);

  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  
  // EXPECT_EQ(this->blob_top_label_->cpu_data()[0], 0);
  // multi-label
  for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
    EXPECT_EQ(this->blob_top_label_->cpu_data()[label_id], label_id);
  }

  // cat gray.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);

  // EXPECT_EQ(this->blob_top_label_->cpu_data()[0], 1);
  // multi-label
  for(int label_id = 0; label_id < this->dim_label_; ++label_id) {
    EXPECT_EQ(this->blob_top_label_->cpu_data()[label_id], label_id);
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
