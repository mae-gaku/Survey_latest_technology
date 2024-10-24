# CUDA高速化基礎

### Memory

1. グローバルメモリ：GPU 内で最も大容量かつ低速なメモリ
  - cudaMalloc: 動的にメモリ確保する
  - __device__: 静的にメモリ確保する
  - cudaFree: メモリ解放する

2. シェアードメモリ：同一ブロック内のスレッドが読み書きできるメモリで、高速  グローバルメモリからシェアードメモリにコピーして参照する
  - __shared__ で修飾して宣言


### Code

1. ホストメモリ(CPU)を確保
2. デバイスメモリ(GPU)を確保
3. ホストメモリからデバイスメモリへデータをコピーする
4. ホストからカーネル関数を実行する
5. 実行結果をデバイスメモリからホストメモリへコピーする


**※どの部分をカーネル関数にしてGPUに処理させるかを考える**
- GPU を使って処理を高速化するには、「forループを外して並列に処理させる」ことが基本的な方針だそう。


```
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <Windows.h>

#define THREAD_NUM_X 32
#define THREAD_NUM_Y 4
#define USE_GPU
#define _TIME

// ホスト関数から呼び出すためglobal で修飾し、戻り値は void
// カーネル関数で実行したい処理を書く
__global__ void ArraySum(int *input1, int* input2, int *output, int width, int height) {

    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    int index = index_y * width + index_x;

    // 行列のサイズを超えたらreturn
    if (width * height <= index) return;

    output[index] = input1[index] + input2[index];
}

void main() {

#ifdef _TIME
    LARGE_INTEGER    frequency, timer_start, timer_end;
    double          theCompressTime;
#endif

    int Width = 1024;
    int Height = 1024;
    int ArraySize = Width * Height;        // 配列のサイズ
    int *Input1, *Input2, *Output;        // ホストメモリ
    int *Input1_d, *Input2_d, *Output_d;    // デバイスメモリ

    // ホストメモリの確保
    Input1 = (int*)malloc(ArraySize * sizeof(int));
    Input2 = (int*)malloc(ArraySize * sizeof(int));
    Output = (int*)malloc(ArraySize * sizeof(int));

    // デバイスメモリの確保(GPU):cudaMalloc()
    // ホストメモリと区別がつくようにデバイスメモリの変数には _d をつけとく。
    cudaMalloc(&Input1_d, ArraySize * sizeof(int));
    cudaMalloc(&Input2_d, ArraySize * sizeof(int));
    cudaMalloc(&Output_d, ArraySize * sizeof(int));

    // 入力データの初期化
    for (int i = 0; i < ArraySize; i++) {
        Input1[i] = 1; // すべての要素を1に設定
        Input2[i] = 2; // すべての要素を2に設定
    }

    // デバイスにデータをコピー(CPU側のデータをGPU側にコピーしている):cudaMemcpy()
    // 第2引数のデータを第1引数にコピーする。ex:Input1⇒Input1_d
    cudaMemcpy(Input1_d, Input1, ArraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Input2_d, Input2, ArraySize * sizeof(int), cudaMemcpyHostToDevice);

    // dim3型という3次元の変数を使って指定(x、y、z。z方向はスレッド数が少ないこともあり使われないことが多い)
    // スレッド数に迷ったら32の倍数を設定する
    dim3 Block(std::ceil(Width / THREAD_NUM_X), std::ceil(Height / THREAD_NUM_Y));
    dim3 Thread(THREAD_NUM_X, THREAD_NUM_Y);

#ifdef _TIME
    QueryPerformanceCounter(&timer_start);
#endif

    // カーネルの実行
    // 関数名 <<< ブロック数, スレッド数 >>> (引数);
    ArraySum << <Block, Thread >> > (Input1_d, Input2_d, Output_d, Width, Height);

#ifdef _TIME
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&timer_end);
    QueryPerformanceFrequency(&frequency);
    theCompressTime = (double)(timer_end.QuadPart - timer_start.QuadPart) / (double)(frequency.QuadPart);
    printf("Time : %lf ms\n", theCompressTime * 1000);
#endif

#ifdef _DEBUG
    // エラーチェック
    // cudaDeviceSynchronize()はバイスの同期を行う関数。前のタスクのうち1つでも失敗した場合はエラーを返す
    // cudaDeviceSynchronize() でエラーコードを取得し、cudaGetErrorString() でエラーメッセージに変換するのが定石
    cudaError_t ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(ret));
    }
#endif

    // 結果をホストにコピー
    // カーネル関数の結果をホストで参照したい場合は、ホストメモリにコピー
    cudaMemcpy(Output, Output_d, ArraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // メモリの解放
    free(Input1); free(Input2); free(Output);
    cudaFree(Input1_d); cudaFree(Input2_d); cudaFree(Output_d);

    return 0;
}

```

- cudaMalloc のオーバーヘッドは大きい
　 cudaMalloc はプログラムの途中で何度も呼び出さない方がパフォーマンスが安定する。  
   メモリはどこか1か所で確保しておいたり、ワーク用のメモリを複数取っておいて、それらを管理しながら使いまわすという方法とかがいいみたい。

- カーネル関数内にif文、for文を入れると遅い
   条件分岐によって異なる実行パスを取るスレッドが存在する場合、GPUはそれぞれのパスを順番に実行する必要があります。その結果、全てのスレッドが同じ命令を同時に実行できる場合に比べて、実行時間が長くなる。
   条件分岐したい場合は、条件によって実行するカーネル関数を変えるなどの工夫を行う必要。

- メモリ転送は最小限にする
   データをデバイスに置いたまますべての処理ができなくホストに転送しなければいけない場面もある。
   メモリ転送は同期的な処理なので多用すると遅くなります。(サイズにもよりますが、1ms～2ms かかる)
   メモリ転送しなくて良いように処理を見直す、非同期でコピーできる **cudaMemcpyAsync** を使うなどの工夫が必要です。




## 参考資料

- 