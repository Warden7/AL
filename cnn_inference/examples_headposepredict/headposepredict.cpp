// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Net squeezenet;

	#if 0
		squeezenet.load_param_bin("ncnn.proto.bin");
	#else
		squeezenet.load_param("ncnn.proto");
	#endif
	squeezenet.load_model("ncnn.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 64, 64);

	const float mean_vals[3] = {127.f, 127.f, 127.f};
	in.substract_mean_normalize(mean_vals, 0);

	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_light_mode(true);

	ex.input("Placeholder_1", in);

	ncnn::Mat out;
	ex.extract("Tanh_9", out);

	cls_scores.resize(out.w);
	for (int j=0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
		fprintf(stderr, "out = %f\n", out[j]*100.0);
	}

	fprintf(stderr, "out.dims = %d\n", out.dims);
	fprintf(stderr, "out.w = %d\n", out.w);
	fprintf(stderr, "out.h = %d\n", out.h);
	fprintf(stderr, "out.c  = %d\n", out.c);
	fprintf(stderr, "out.elemsize = %d\n", out.elemsize);

	return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    print_topk(cls_scores, 3);

    return 0;
}

