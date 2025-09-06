% 粒子群优化（PSO）：第四问 - 三架无人机（FY1/2/3）各投一弹，最大化总遮挡
% 目标函数：在全局时间轴上判断任一时刻是否被任一云团遮挡，计算总遮挡时长；
% 奖励：使用云团爆炸位置到导弹轨迹的距离（越近奖励越大）

clear; clc;

%% 约束与场景常量
% 速度大小与方位角
SPEED_MIN = 70; SPEED_MAX = 140;     % s ∈ [70,140]
THETA_MIN = -pi; THETA_MAX = pi;     % theta ∈ [-pi, pi]
VM_kappa = 10;                        % 初始方位角冯米塞斯浓度

% 时间约束
T_MAX = 20;
T_SUM_MAX = 20 * (1 - 1e-6);         % 总时间上限（防数值卡边）

% 目标函数仿真选项
sim_opts = struct('tSimEnd', 20, 'g', 9.8);

% 目标/导弹/云团参数（与Q1一致，按需调整）
v_cloud   = 3;                       % 云团垂直坠落速度
r_cloud   = 10;                      % 云团半径
r_target  = 7;                       % 目标半径
h_target  = 10;                      % 目标高度
pos_target = [0, 200, 0];
pos_M1     = [20000, 0, 2000];
v_M       = 300;                     % 导弹速度
vv_M1     = v_M * (pos_target - pos_M1) / norm(pos_target - pos_M1);

% 三架无人机初始位置（请按实际题面调整）
pos_FY1    = [17800, 0, 1800];
pos_FY2    = [17600, -500, 1800];
pos_FY3    = [18000,  600, 1800];

%% 奖赏机制参数（退火）
REWARD_MAX_ITER_FRAC = 0.5;         % 前50%迭代使用明显奖励
REWARD_WEIGHT_INIT   = 1;          % 初始奖赏权重
DIST_SCALE           = 300;          % 距离尺度（m），越小越强调接近导弹轨迹

%% 粒子群参数与搜索边界
% 变量（12维）：[s1 th1 t1 dt1  s2 th2 t2 dt2  s3 th3 t3 dt3]
nVar = 12;
VarMin = [SPEED_MIN, THETA_MIN, 0, 0,  SPEED_MIN, THETA_MIN, 0, 0,  SPEED_MIN, THETA_MIN, 0, 0];
VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX,  SPEED_MAX, THETA_MAX, T_MAX, T_MAX,  SPEED_MAX, THETA_MAX, T_MAX, T_MAX];

nPop = 120;              % 群体规模
MaxIt = 60;             % 最大迭代次数
w = 1.2;                % 惯性权重
wDamp = 0.99;            % 惯性权重衰减
c1 = 1.6;                % 个体学习因子
c2 = 2.0;                % 群体学习因子

% 速度上限（按搜索区间比例）
VelMax = 0.25 * (VarMax - VarMin);
VelMin = -VelMax;

%% 目标函数封装（最小化：-遮挡时长 - 奖励项）
obj = @(x, it) objective_q4(x, sim_opts, ...
    v_cloud, r_cloud, r_target, h_target, pos_target, pos_M1, vv_M1, ...
    pos_FY1, pos_FY2, pos_FY3, ...
    REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT, DIST_SCALE, it, MaxIt);

%% 粒子结构体
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost     = [];
empty_particle.Duration = [];
empty_particle.Reward   = 0;
empty_particle.Best = empty_particle;

% 初始化群体
particle  = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf; GlobalBest.Duration = -inf; GlobalBest.Reward = -inf; GlobalBest.Position = [];

for i = 1:nPop
    particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);
    % 三个无人机的theta围绕其指向目标的方位角采样
    target_theta_1 = atan2(pos_target(2)-pos_FY1(2), pos_target(1)-pos_FY1(1));
    target_theta_2 = atan2(pos_target(2)-pos_FY2(2), pos_target(1)-pos_FY2(1));
    target_theta_3 = atan2(pos_target(2)-pos_FY3(2), pos_target(1)-pos_FY3(1));
    particle(i).Position(2)  = wrapToPi(target_theta_1 + randVonMises(0, VM_kappa, 1));
    particle(i).Position(6)  = wrapToPi(target_theta_2 + randVonMises(0, VM_kappa, 1));
    particle(i).Position(10) = wrapToPi(target_theta_3 + randVonMises(0, VM_kappa, 1));

    particle(i).Velocity = zeros(1, nVar);
    [J, dur, rew] = obj(particle(i).Position, 1);
    particle(i).Cost = J; particle(i).Duration = dur; particle(i).Reward = rew;
    particle(i).Best = particle(i);
    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end

% 记录收敛
BestCost = nan(MaxIt,1); BestDuration = nan(MaxIt,1); BestReward = nan(MaxIt,1);

% 停滞检测（多样性注入）
stallThreshold = 18; improveTol = 1e-6; stallCounter = 0; prevBestCost = inf;

%% 迭代优化
for it = 1:MaxIt

    for i = 1:nPop
        r1 = rand(1, nVar); 
        r2 = rand(1, nVar);
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*r1.*(particle(i).Best.Position - particle(i).Position) ...
            + c2*r2.*(GlobalBest.Position - particle(i).Position);
        particle(i).Velocity = min(max(particle(i).Velocity, VelMin), VelMax);
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        particle(i).Position = min(max(particle(i).Position, VarMin), VarMax);
        % 角度wrap
        particle(i).Position([2,6,10]) = arrayfun(@wrapToPi, particle(i).Position([2,6,10]));

        % 重新评估
        [J, dur, rew] = obj(particle(i).Position, it);
        particle(i).Cost = J; 
        particle(i).Duration = dur; 
        particle(i).Reward = rew;

        % 个体最优
        if J < particle(i).Best.Cost
            particle(i).Best = particle(i);
        end

        % 全局最优（前期考虑奖励并列）
        if it <= REWARD_MAX_ITER_FRAC * MaxIt
            if particle(i).Best.Cost < GlobalBest.Cost || ...
               (abs(particle(i).Best.Cost - GlobalBest.Cost) < 1e-6 && ...
                particle(i).Best.Reward > GlobalBest.Reward)
                GlobalBest = particle(i).Best;
            end
        else
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
    end

    BestCost(it) = GlobalBest.Cost; 
    BestDuration(it) = GlobalBest.Duration; 
    BestReward(it) = GlobalBest.Reward;

    % 停滞检测与注入
    if it == 1 || (BestCost(it) < prevBestCost - improveTol)
        prevBestCost = BestCost(it); stallCounter = 0;
    else
        stallCounter = stallCounter + 1;
    end
    if stallCounter >= stallThreshold
        numReseed = max(1, round(0.10 * nPop));
        costs = [particle.Cost]; [~, idxSorted] = sort(costs, 'descend'); reseedIdx = idxSorted(1:numReseed);
        for k = 1:numReseed
            j = reseedIdx(k); pos = particle(j).Position;
            % 时间维（t,dt）高斯扰动（标准差按各维度范围的12%）
            timeIdx = [3,4, 7,8, 11,12];
            timeRange = VarMax(timeIdx) - VarMin(timeIdx);
            timeStd = 0.12 * timeRange;
            pos(timeIdx) = pos(timeIdx) + randn(1,6).*timeStd;
            pos(timeIdx) = VarMin(timeIdx) + rand(1,6).*(VarMax(timeIdx)-VarMin(timeIdx));
            % s/theta 高斯扰动
            thetaStd = deg2rad(20);
            pos([2,6,10]) = arrayfun(@(th) wrapToPi(th + randn*thetaStd), pos([2,6,10]));
            sStd = 0.08*(SPEED_MAX-SPEED_MIN);
            pos([1,5,9]) = min(max(pos([1,5,9]) + randn(1,3)*sStd, SPEED_MIN), SPEED_MAX);
            % 夹紧
            pos = min(max(pos, VarMin), VarMax);
            particle(j).Position = pos;
            particle(j).Velocity = zeros(1, nVar);
            [Jr, durR, rewardR] = obj(particle(j).Position, it);
            particle(j).Cost = Jr; 
            particle(j).Duration = durR; 
            particle(j).Reward = rewardR;

            particle(j).Best.Position = particle(j).Position;
            particle(j).Best.Cost = Jr;
            particle(j).Best.Duration = durR;
            particle(j).Best.Reward = rewardR;
        end
        stallCounter = 0;
    end
    % 惯性权重衰减
    w = w * wDamp;
    fprintf('Iter %3d | BestDuration = %.6f s | Cost = %.6f | Reward = %.6f\n', it, GlobalBest.Duration, GlobalBest.Cost, GlobalBest.Reward);
end

%% 输出最优解
xbest = GlobalBest.Position;
[s1,th1,t1,dt1, s2,th2,t2,dt2, s3,th3,t3,dt3] = deal(xbest(1),xbest(2),xbest(3),xbest(4), xbest(5),xbest(6),xbest(7),xbest(8), xbest(9),xbest(10),xbest(11),xbest(12));
[vx1,vy1] = deal(s1*cos(th1), s1*sin(th1));
[vx2,vy2] = deal(s2*cos(th2), s2*sin(th2));
[vx3,vy3] = deal(s3*cos(th3), s3*sin(th3));

fprintf('\n最优解：\n');
fprintf('  FY1: s=%.3f, th=%.3f, t=%.3f, dt=%.3f\n', s1, th1, t1, dt1);
fprintf('  FY2: s=%.3f, th=%.3f, t=%.3f, dt=%.3f\n', s2, th2, t2, dt2);
fprintf('  FY3: s=%.3f, th=%.3f, t=%.3f, dt=%.3f\n', s3, th3, t3, dt3);

totalDuration = GlobalBest.Duration;
fprintf('\n  最优遮挡总时长 = %.6f s\n', totalDuration);

%% 子函数：目标函数
function [J, duration, reward] = objective_q4(x, sim_opts, ...
    v_cloud, r_cloud, r_target, h_target, pos_target, pos_M1, vv_M1, ...
    pos_FY1, pos_FY2, pos_FY3, ...
    reward_frac, reward_init, dist_scale, it, MaxIt)

    % 解包变量
    s1=x(1); th1=x(2); t1=x(3); dt1=x(4);
    s2=x(5); th2=x(6); t2=x(7); dt2=x(8);
    s3=x(9); th3=x(10); t3=x(11); dt3=x(12);
    vv1 = [s1*cos(th1), s1*sin(th1), 0];
    vv2 = [s2*cos(th2), s2*sin(th2), 0];
    vv3 = [s3*cos(th3), s3*sin(th3), 0];

    % 计算三枚云团的爆炸中心（各自引爆瞬间）
    pos_throw_1 = pos_FY1 + t1 * vv1; pos_bao_1 = pos_throw_1 + dt1 * vv1; pos_bao_1(3) = pos_bao_1(3) - 0.5 * sim_opts.g * (dt1^2);
    pos_throw_2 = pos_FY2 + t2 * vv2; pos_bao_2 = pos_throw_2 + dt2 * vv2; pos_bao_2(3) = pos_bao_2(3) - 0.5 * sim_opts.g * (dt2^2);
    pos_throw_3 = pos_FY3 + t3 * vv3; pos_bao_3 = pos_throw_3 + dt3 * vv3; pos_bao_3(3) = pos_bao_3(3) - 0.5 * sim_opts.g * (dt3^2);

    % 构造全局时间轴多云团遮挡
    spheres = struct('startTime', {}, 'center0', {}, 'vel', {}, 'radius', {});
    spheres(1).startTime = t1+dt1; spheres(1).center0 = pos_bao_1; spheres(1).vel = [0,0,-v_cloud]; spheres(1).radius = r_cloud;
    spheres(2).startTime = t2+dt2; spheres(2).center0 = pos_bao_2; spheres(2).vel = [0,0,-v_cloud]; spheres(2).radius = r_cloud;
    spheres(3).startTime = t3+dt3; spheres(3).center0 = pos_bao_3; spheres(3).vel = [0,0,-v_cloud]; spheres(3).radius = r_cloud;

    [duration, cloud_intervals] = compute_occlusion_multi(0, sim_opts.tSimEnd, ...
        pos_M1, vv_M1, pos_target, r_target, h_target, spheres);

    % 可以计算每个云团的有效遮挡时长
    cloud_durations = zeros(1, 3);
    for i = 1:3
        if ~isempty(cloud_intervals{i})
            intervals = cell2mat(cloud_intervals{i});
            cloud_durations(i) = sum(intervals(:,2) - intervals(:,1));
        end
    end

    % 输出每个云团的贡献
    fprintf('云团1遮挡时长: %.2f秒\n', cloud_durations(1));
    fprintf('云团2遮挡时长: %.2f秒\n', cloud_durations(2));
    fprintf('云团3遮挡时长: %.2f秒\n', cloud_durations(3));
    fprintf('总有效遮挡时长: %.2f秒\n', duration);

    % 奖励：云团爆炸点到导弹轨迹直线的距离（越近越好）
    d1 = point_to_line_distance(pos_bao_1, pos_M1, vv_M1);
    d2 = point_to_line_distance(pos_bao_2, pos_M1, vv_M1);
    d3 = point_to_line_distance(pos_bao_3, pos_M1, vv_M1);
    % 使用生存函数型奖励（归一化到0~1之间）
    reward_each = exp(-[d1, d2, d3]/dist_scale);
    reward = mean(reward_each);
    
    % 迭代退火权重
    reward_weight = max(0, reward_init*(1 - it/(reward_frac*MaxIt)));

    % 目标函数（最小化）
    J = -duration - reward_weight*reward;
end

%% 子函数：多云团全局遮挡
function [duration, cloud_intervals] = compute_occlusion_multi(tStart, tEnd, pos_M1, vv_M1, pos_target, r_target, h_target, spheres)
    % 计算多云团遮挡总时长及各云团的遮挡区间
    % 输出:
    %   duration: 总遮挡时长(秒)
    %   cloud_intervals: 元胞数组，每个元素包含一个云团的遮挡时间区间 [start, end]

    % 初始化输出
    duration = 0;
    cloud_intervals = cell(length(spheres), 1);
    
    % 计算每个云团的遮挡区间
    all_intervals = {};
    
    for i = 1:length(spheres)
        sphere = spheres(i);
        % 计算当前云团的遮挡区间
        intervals = computeOcclusionSimple(tStart, tEnd, ...
            pos_M1, vv_M1, pos_target, r_target, h_target, ...
            sphere.center0, sphere.vel, sphere.radius, sphere.startTime);
        
        % 保存该云团的遮挡区间
        cloud_intervals{i} = intervals;
        
        % 添加到所有区间列表
        if ~isempty(intervals)
            all_intervals = [all_intervals; intervals];
        end
    end
    
    % 合并所有区间计算总遮挡时长
    if ~isempty(all_intervals)
        merged_intervals = merge_intervals(vertcat(all_intervals{:}));
        duration = sum(merged_intervals(:,2) - merged_intervals(:,1));
    end
end

function merged = merge_intervals(intervals)
    % 合并重叠的时间区间
    if isempty(intervals)
        merged = [];
        return;
    end
    
    % 按开始时间排序
    [~, idx] = sort(intervals(:,1));
    intervals = intervals(idx,:);
    
    merged = intervals(1,:);
    for i = 2:size(intervals, 1)
        current = intervals(i,:);
        last = merged(end,:);
        
        % 检查是否重叠
        if current(1) <= last(2)
            % 重叠，合并区间
            merged(end,2) = max(last(2), current(2));
        else
            % 不重叠，添加新区间
            merged = [merged; current];
        end
    end
end

function tc = refine(ta, tb, occA, fOcc, tol)
    while (tb - ta) > tol
        tm = 0.5*(ta+tb);
        if fOcc(tm) == occA, ta = tm; else, tb = tm; end
    end
    tc = tb;
end

%% 子函数：点到直线距离
function d = point_to_line_distance(p, p0, v)
    % p 到 直线 L: p0 + t*v 的最短距离
    d = norm(cross(v, (p - p0))) / norm(v);
end

%% 子函数：von Mises 采样（仅用于微扰角度，mu=0时即对称扰动）
function theta = randVonMises(mu, kappa, n)
    if kappa < 1e-8
        theta = (rand(n,1)*2*pi - pi) + mu; theta = mod(theta + pi, 2*pi) - pi; return;
    end
    a = 1 + sqrt(1 + 4*kappa^2); b = (a - sqrt(2*a)) / (2*kappa); r = (1 + b^2) / (2*b);
    theta = zeros(n,1);
    for ii = 1:n
        while true
            u1 = rand; u2 = rand; u3 = rand; z = cos(pi*u1); f = (1 + r*z) / (r + z); c = kappa * (r - f);
            if (u2 < c*(2 - c)) || (log(u2) <= (c - 1)), break; end
        end
        signv = 1; if u3 - 0.5 < 0, signv = -1; end
        theta(ii) = mu + signv * acos(f);
    end
    theta = mod(theta + pi, 2*pi) - pi;
end
