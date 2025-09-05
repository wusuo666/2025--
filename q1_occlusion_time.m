function [totalDuration, intervals] = q1_occlusion_time(vx_FY1, vy_FY1, t_throw, t_explode, opts)
% 目标函数（Question 1）：给定FY1速度向量分量与投弹/引爆时刻，计算遮挡总时长
% 输入：
%   vx_FY1, vy_FY1 : FY1在投放阶段的水平速度分量（m/s）
%   t_throw        : 投弹时刻（s）
%   t_explode      : 引爆时刻（s）- 从投弹算起到引爆的时间间隔
%   opts           : 可选结构体：
%       .tSimEnd   : 仿真结束时间（默认 20 s）
%       .dt        : 固定步长（默认 0.01 s）
%       .g         : 重力加速度（默认 9.8 m/s^2）
% 输出：
%   totalDuration  : 遮挡总时长（s）
%   intervals      : 遮挡区间 [t_begin, t_end]（相对引爆时刻的后续时间）
%
% 说明：
% - 内部直接调用 computeOcclusionSimple（自适应步长）和 isCylinderBlockedBySphere。
% - 使用题面给定的默认常量与初始位置，可在需要时根据实际题目修改。

    if nargin < 5, opts = struct(); end
    if ~isfield(opts, 'tSimEnd'), opts.tSimEnd = 20; end
    if ~isfield(opts, 'dt'),       opts.dt = 0.01; end
    if ~isfield(opts, 'g'),        opts.g = 9.8; end

    % ---------------- 场景常量（题面示例，按需调整） ----------------
    v_cloud   = 3;     % 云团坠落速度
    r_cloud   = 10;    % 云团半径
    v_M       = 300;   % 导弹速度（M1）
    r_target  = 7;     % 目标半径
    h_target  = 10;    % 目标高度

    % 关键位置（题面示例）
    pos_target = [0, 200, 0];
    pos_fake   = [0, 0, 0];
    pos_M1     = [20000, 0, 2000];
    pos_FY1    = [17800, 0, 1800];

    % ---------------- 由输入构造动力学初值 ----------------
    % FY1 水平速度
    vv_FY1 = [vx_FY1, vy_FY1, 0];

    % 导弹M1速度（指向假目标）
    vv_M1 = v_M * (pos_fake - pos_M1) / norm(pos_fake - pos_M1);

    % 投放点位置（FY1飞行 t_throw 后）
    pos_throw = pos_FY1 + t_throw * vv_FY1;

    % 爆炸中心位置（从投放点沿FY1方向飞行 t_explode，再考虑重力下落）
    pos_bao = pos_throw + t_explode * vv_FY1;
    pos_bao(3) = pos_bao(3) - 0.5 * opts.g * (t_explode^2);

    % 云团速度（仅垂直向下）
    vv_bao = [0, 0, -v_cloud];

    % 观察者（M1）在引爆瞬间的位置（从初始到 t_throw+t_explode 按匀速）
    pos_M1_bao = pos_M1 + (t_throw + t_explode) * vv_M1;

    % ---------------- 组装遮挡仿真初始状态 ----------------
    init = struct();
    init.observerPos    = pos_M1_bao;
    init.observerVel    = vv_M1; % 之后继续按相同速度飞行
    init.cylinderCenter = [pos_target(1), pos_target(2), h_target/2];
    init.cylinderVel    = [0, 0, 0];
    init.cylinderRadius = r_target;
    init.cylinderHeight = h_target;
    init.cylinderDir    = [0, 0, 1];
    init.sphereCenter   = pos_bao;
    init.sphereVel      = vv_bao;
    init.sphereRadius   = r_cloud;

    % ---------------- 自适应步长计算遮挡 ----------------
    t0 = 0;
    t1 = opts.tSimEnd;
    [totalDuration, intervals] = computeOcclusionSimple(t0, t1, init);
end 