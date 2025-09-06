function pos_uav = position_bao(vx_FY1, vy_FY1, t_throw, t_explode, pos_FY1_init)
% 计算FY1在烟雾弹引爆时刻的空间位置
% 输入参数：
%   vx_FY1, vy_FY1 : FY1在投放阶段的水平速度分量（m/s）
%   t_throw        : 投弹时刻（s）
%   t_explode      : 引爆时刻（s）- 从投弹算起到引爆的时间间隔
%   pos_FY1_init   : FY1初始位置 [x, y, z]
% 输出参数：
%   pos_uav        : FY1在 (t_throw + t_explode) 时刻的位置 [x, y, z]
%
% 说明：
% - 假设FY1在投放阶段保持恒定水平速度 [vx_FY1, vy_FY1, 0]，高度不变。
% - 因此FY1的 z 分量保持为初始高度 pos_FY1_init(3)。

    % 输入检查
    if nargin < 5
        error('需要提供 vx_FY1, vy_FY1, t_throw, t_explode, pos_FY1_init 五个参数。');
    end
    if numel(pos_FY1_init) ~= 3
        error('pos_FY1_init 必须为 1x3 向量 [x, y, z]。');
    end

    % FY1 水平速度向量
    vv_FY1 = [vx_FY1, vy_FY1, 0];

    % 从初始时刻到引爆时刻的总飞行时间
    t_total = t_throw + t_explode;

    % FY1 在引爆时刻的位置（保持高度不变）
    pos_uav = pos_FY1_init + t_total * vv_FY1;
end 