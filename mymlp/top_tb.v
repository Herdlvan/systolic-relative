`timescale 1ns/1ps

module top_tb;

    parameter DATA_WIDTH = 8;

    reg clk;
    reg reset;
    reg [9:0] M_index, N_index, K_index;
    reg read_enable;
    wire [DATA_WIDTH*387-1:0] softmax_out;
    wire done;
    wire [8:0] predicted_index;

    // 实例化待测模块
    top #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .clk(clk),
        .reset(reset),
        .M_index(M_index),
        .N_index(N_index),
        .K_index(K_index),
        .read_enable(read_enable),
        .softmax_out(softmax_out),
        .done(done),
        .predicted_index(predicted_index)
    );

    // 时钟生成
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz

    initial begin
        // 波形dump
        $dumpfile("wave.vcd");
        $dumpvars(0, top_tb);

        // 初始化
        reset = 1;
        read_enable = 0;
        M_index = 0;
        N_index = 0;
        K_index = 0;

        // 等待一段时间释放复位
        #20;
        reset = 0;

        // 激活读取
        #10;
        read_enable = 1;

        // 输入不同的index组合进行测试
        #10;
        M_index = 10; N_index = 20; K_index = 30;
        #10;
        M_index = 100; N_index = 200; K_index = 300;
        #10;
        M_index = 1; N_index = 2; K_index = 3;

        // 等待softmax done信号
        wait(done);

        // 打印输出
        $display("Predicted index: %d", predicted_index);
        $display("Softmax out (first 8 bits): %h", softmax_out[7:0]);

        // 结束仿真
        #20;
        $finish;
    end

endmodule