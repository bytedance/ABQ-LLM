// Copyright 2024 ByteDance and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iterator> // for generating row_indices
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;


void print_layout(int row, int col){
    std::vector<std::vector<std::string>> matrix(row, std::vector<std::string>(col, "--"));
    for (size_t lane_rank = 0; lane_rank < 32; lane_rank++)
    {
        /* code */
        // // // ShapeBase<16, 16, 8 > a  fragment_a_colmajor  trans:True
        // int row_index  = lane_rank & 8;
        // int col_index  = lane_rank & 7;

        // // // // ShapeBase<16, 16, 8 > a fragment_a_rowmajor trans:False
        // int row_index  = lane_rank & 15;
        // int col_index  = 0;

        // // ShapeBase<16, 16, 16> a     fragment_a_colmajor  trans:True
        // int row_index = lane_rank & 8;
        // int col_index = (lane_rank & 7) + ((lane_rank & 16)>>1);

        // // // // ShapeBase<16, 16, 16 > a fragment_a_rowmajor  trans:False
        // int row_index  = (lane_rank & 15);
        // int col_index  = ((lane_rank & 16)>>1);

        // // ShapeBase<16, 16, 8 > b     fragment_b_rowmajor  trans:True
        // int row_index  = lane_rank & 7;
        // int col_index  = lane_rank & 8;

        // // ShapeBase<16, 16, 8 > b      fragment_b_colmajor  trans:False
        // int row_index  = 0;
        // int col_index  = lane_rank & 15;

        // // ShapeBase<16, 16, 16> b      fragment_b_rowmajor  trans:True
        // int row_index  = (lane_rank & 15);
        // int col_index  = ((lane_rank & 16)>>1);

        // // // ShapeBase<16, 16, 16> b   fragment_b_colmajor  trans:False
        // int row_index  = (lane_rank & 8);
        // int col_index  = (lane_rank & 7) + ((lane_rank & 16)>>1);
        
        // // store_matrix_sync ShapeBase<16, 16, 8 >
        // int row_index  = (lane_rank / 4);
        // int col_index  = (lane_rank & 3) *2; 


        // // < 750 
        // int row_index = lane_rank & 3;
        // int col_index = ((lane_rank & 4)<<1) + ((lane_rank & 16) >> 2);

        int row_index = lane_rank & 3;
        int col_index = ((lane_rank & 4)<<2) + ((lane_rank & 16) >> 1);

        std::ostringstream format_value;
        format_value << std::dec << std::setw(2) << std::setfill(' ') << lane_rank;
        std::cout << "row_index:" << row_index << " col_index:" << col_index << " value:" << format_value.str() << std::endl;
        matrix[row_index][col_index] = format_value.str();
    }
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, const char** argv) {
    print_layout(32,16);
    return 0;
}