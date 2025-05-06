#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <bitset>

std::vector<float> generate_boxes(int num_boxes, int num_cluster) {
    constexpr float image_width = 1920.f;
    constexpr float image_height = 1080.f;
    auto boxes = std::vector<float>(num_boxes*5, 0.f);
    auto num_per_cluster = num_boxes / num_cluster;

    for (int i=0; i<num_cluster; ++i) {
        auto const center_x = image_width * (i + 1) / (num_cluster + 2);
        auto const center_y = image_height * (i + 1) / (num_cluster + 2);

        for (int j=0; j<num_per_cluster; ++j) {
            std::random_device rd;
            auto rng = std::default_random_engine(rd());
            auto dist = std::uniform_real_distribution<float>(0., 1.);

            auto ri = dist(rng);

            auto box_x1 = center_x - 50 - ri;
            auto box_x2 = center_x + 50 + ri;
            auto box_y1 = center_y - 50 - ri;
            auto box_y2 = center_y + 50 + ri;
            auto box_score = 1.f; // TODO: 这里假设分数都一样

            boxes[i*num_per_cluster*5+j*5] = box_x1;
            boxes[i*num_per_cluster*5+j*5+1] = box_y1;
            boxes[i*num_per_cluster*5+j*5+2] = box_x2;
            boxes[i*num_per_cluster*5+j*5+3] = box_y2;
            boxes[i*num_per_cluster*5+j*5+4] = box_score;
        }
    }

    return boxes;
}

__device__ inline bool calculate_iou(
    float x11, float y11, float x12, float y12,
    float x21, float y21, float x22, float y22,
    float const threshold
) {
    float x_i1 = fmaxf(x11, x21);
    float x_i2 = fminf(x12, x22);
    float y_i1 = fmaxf(y11, y21);
    float y_i2 = fminf(y12, y22);

    float width = fmaxf((x_i2 - x_i1), 0.f);
    float height = fmaxf((y_i2 - y_i1), 0.f);

    float inter = width * height;
    float area = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21);

    return inter / (area - inter) > threshold;
}

__global__ void calculate_iou_matrix(uint64_t *iou_matrix, float const *__restrict boxes, int const num_boxes, float const threshold) {
    __shared__ float shared_boxes[64][5];
    if (blockIdx.y > blockIdx.x) return;

    // 以防 num_boxes 不能被 64 整除
    int row_size = min(num_boxes - blockIdx.y * blockDim.x, blockDim.x);
    int col_size = min(num_boxes - blockIdx.x * blockDim.x, blockDim.x);

    // 每个线程算 cur_thread_box_idx 和 这个线程块其他线程的 cur_thread_box_idx 对应 box 的 iou
    // 比如这个线程的 cur_thread_box_idx = 0 * 64 + 0 = 0, 这个线程块所有 cur_thread_box_idx 的取值是 [0, 63]
    int cur_box_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int target_block_box_start_idx = blockIdx.x * blockDim.x;
    int target_thread_box_idx = target_block_box_start_idx + threadIdx.x;
    float const *target_box = boxes + target_thread_box_idx * 5;
    float const *cur_box = boxes + cur_box_idx * 5;

    if (threadIdx.x < col_size) {
        // 把 box 装到 shared_memory 里
        shared_boxes[threadIdx.x][0] = target_box[0];
        shared_boxes[threadIdx.x][1] = target_box[1];
        shared_boxes[threadIdx.x][2] = target_box[2];
        shared_boxes[threadIdx.x][3] = target_box[3];
        shared_boxes[threadIdx.x][4] = target_box[4];
    }
    __syncthreads();

    // 开算, 由于 iou(i,j) == iou(j,i), 我们只算上三角的
    // TODO: 有很多 64, magic number 并不好
    if (threadIdx.x < row_size) {
        float x11 = cur_box[0], y11 = cur_box[1], x12 = cur_box[2], y12 = cur_box[3];
        uint64_t t = 0;
        int i = blockIdx.x == blockIdx.y ? threadIdx.x + 1: 0;

        for (; i < col_size; ++i) {
            float x21 = shared_boxes[i][0], y21 = shared_boxes[i][1];
            float x22 = shared_boxes[i][2], y22 = shared_boxes[i][3];

            bool iou_large_enough = calculate_iou(
                x11, y11, x12, y12,
                x21, y21, x22, y22,
                threshold
            );

            if (false && threadIdx.x == 0 && i%32 == 1) {
                printf("cur: box[%d] = (%f, %f, %f, %f) \n", cur_box_idx, x11, y11, x12, y12);
                printf("tar: box[%d] = (%f, %f, %f, %f) \n", i, x21, y21, x22, y22);
                printf("iou(%d, %d) = %d \n", cur_box_idx, i, (int)iou_large_enough);
            }

            if (iou_large_enough)
                t |= (1ULL << (63 - i));
        }

        // iou_matrix.shape = [num_boxes, num_boxes / 64]
        // gridDim.x = num_boxes / 64
        iou_matrix[cur_box_idx * gridDim.x + blockIdx.x] = t;
    }
}

int main() {
    constexpr int num_boxes = 64;
    constexpr int num_cluster = 2;
    constexpr float iou_thres = 0.1;

    // 假设排好序了, 并且分数都大于 score_thresh
    auto boxes = generate_boxes(num_boxes, num_cluster);
    float *boxes_cuda; cudaMalloc(&boxes_cuda, sizeof(float)*boxes.size());
    cudaMemcpy(boxes_cuda, boxes.data(), sizeof(float)*boxes.size(), cudaMemcpyDefault);

    constexpr int threads = 64; // sizeof(uint64)
    int blocks = (num_boxes + threads - 1) / threads;

    uint64_t *iou_matrix_cuda; cudaMalloc(&iou_matrix_cuda, sizeof(uint64_t)*num_boxes*blocks);

    calculate_iou_matrix<<<dim3(blocks, blocks, 1), dim3(threads, 1, 1)>>>(iou_matrix_cuda, boxes_cuda, num_boxes, iou_thres);
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    uint64_t *iou_matrix = (uint64_t *)malloc(sizeof(uint64_t)*num_boxes*blocks);
    cudaMemcpy(iou_matrix, iou_matrix_cuda, sizeof(uint64_t)*num_boxes*blocks, cudaMemcpyDefault);

    std::vector<int> kept_index;
    std::vector<bool> remove_index(num_boxes, false);
    for (int i=0; i<num_boxes; ++i) {
        if (!remove_index[i]) {
            kept_index.push_back(i);
            remove_index[i] = true;

            for (int j=i+1; j<num_boxes; ++j) {
                bool iou_ij = iou_matrix[i*blocks + j/threads] & (1ull << (63 - j));
                if (iou_ij)
                    remove_index[j] = true;
            }
        }
    }

    std::cout << "kept index: ";
    for (int i=0; i<kept_index.size(); ++i)
        std::cout << kept_index[i] << " ";
    std::cout << std::endl;

    return 0;
}