#3D_optical_flow
import numpy as np
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid,bilinear_samplere
import matplotlib.pyplot as plt
from core.update import BasicUpdateBlock, SmallUpdateBlock, ScaleflowUpdateBlock, DCUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.corr import CorrpyBlock_3D_optical_flow, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from core.utils.resnet import FPN
import inverse_cuda as inv_flow
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass
#这个在算之前，确保代价在0以上
def inverse_opticalflow(flow2,corr,corrdata):
    b, c , h, w = corr.shape

    corrdata = corrdata.reshape(b,h,w,h,w)

    xl = corr[:,0:1,:,:].floor()
    xr = xl +1
    yu = corr[:, 1:2, :, :].floor()
    yd = yu + 1
    output = torch.zeros((b,h,w,4,5)).cuda()
    output[:,:,:,:,2] = -1000
    p1 = torch.cat([xl,yu],dim=1).permute(0,2,3,1).unsqueeze(3)
    p2 = torch.cat([xr, yu], dim=1).permute(0, 2, 3, 1).unsqueeze(3)
    p3 = torch.cat([xr, yd], dim=1).permute(0, 2, 3, 1).unsqueeze(3)
    p4 = torch.cat([xl, yd], dim=1).permute(0, 2, 3, 1).unsqueeze(3)
    pix_index = torch.cat([p1,p2,p3,p4],dim=3).cuda()

    ans = inv_flow.forward(pix_index.contiguous(), corrdata.cuda().contiguous(), flow2.cuda().contiguous(),output.contiguous())
    ans_s = torch.zeros((b,h,w,5))
    index = torch.argmax(ans[:,:,:,:,2],dim=3,keepdim=True)

    out1 = inv_flow.index(ans.contiguous(),index.cuda().contiguous().float(),ans_s.cuda())
    return out1.permute(0,3,1,2)
def inverse_opticalflow_nearset(flow2,corr):
    b, c , h, w = corr.shape

    xl = corr[:,0:1,:,:].floor()
    xr = xl +1
    yu = corr[:, 1:2, :, :].floor()
    yd = yu + 1
    output = torch.zeros((b,h,w,4,5)).cuda()
    output[:,:,:,:,2] = -1000
    p1 = torch.cat([xl,yu],dim=1).permute(0,2,3,1).unsqueeze(3)
    p2 = torch.cat([xr, yu], dim=1).permute(0, 2, 3, 1).unsqueeze(3)
    p3 = torch.cat([xr, yd], dim=1).permute(0, 2, 3, 1).unsqueeze(3)
    p4 = torch.cat([xl, yd], dim=1).permute(0, 2, 3, 1).unsqueeze(3)
    pix_index = torch.cat([p1,p2,p3,p4],dim=3).cuda()
    corr_index = corr.permute(0, 2, 3, 1).unsqueeze(3).cuda()
    xy = pix_index - corr_index
    corrdata = torch.sqrt(xy[:, :, :, :, 0] * xy[:, :, :, :, 0] + xy[:, :, :, :, 1] * xy[:, :, :, :, 1])
    ans = inv_flow.nearset(pix_index.contiguous(), corrdata.cuda().contiguous(), flow2.cuda().contiguous(),output.contiguous())
    ans_s = torch.zeros((b,h,w,5))
    index = torch.argmax(ans[:,:,:,:,2],dim=3,keepdim=True)

    out1 = inv_flow.index(ans.contiguous(),index.cuda().contiguous().float(),ans_s.cuda())
    return out1.permute(0,3,1,2)
# best_weight raft-kitti_11.pth
def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    h = h/h.sum()
    return h


class RAFT3D(nn.Module):
    def __init__(self, args):
        super(RAFT3D, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 192
            self.context_dim = cdim = 192
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = ScaleflowUpdateBlock(self.args, hidden_dim=hdim)
        self.dc_block = DCUpdateBlock(self.args,hdim)
        # blur1 = cv2.resize(blur1, (int(w * (xita / 2)), int(h * (xita / 2))))


        xita = (2 ** 0.5) - 1  # 0.75 + 1
        self.delta12 = 1.5/2
        kernel12 = gaussian2D([5, 5], sigma=(xita, xita))
        xita = (2 ** 1) - 1  # 1 + 1
        self.delta14 = 0.5
        kernel14 = gaussian2D([5, 5], sigma=(xita, xita))
        xita = (2 ** 0.25) - 1  # 0.25 + 1
        self.delta1 = 1.75 / 2
        kernel1 = gaussian2D([5, 5], sigma=(xita, xita))
        xita = (2 ** 0.5) - 1  # 0.5 + 1
        self.delta2 = 1.5 / 2
        kernel2 = gaussian2D([5, 5], sigma=(xita, xita))
        xita = (2 ** 0.75) - 1  # 0.75 + 1
        self.delta3 = 1.25 / 2
        kernel3 = gaussian2D([5, 5], sigma=(xita, xita))
        xita = (2 ** 1) - 1  # 1 + 1
        self.delta4 = 1 / 2
        kernel4 = gaussian2D([5, 5], sigma=(xita, xita))
        #反向流
        kernel = torch.FloatTensor(kernel12).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weight12 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel14).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weight14 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weighto1 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weighto2 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weighto3 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weighto4 = nn.Parameter(data=kernel, requires_grad=False)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    def initialize_exp(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape

        exp = torch.ones((N, H // 8, W // 8)).to(img.device) * 2

        # optical flow computed as difference: flow = coords1 - coords0
        return exp
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    def upsample_exp(self, exp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = exp.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        m = nn.ReplicationPad2d(1)
        up_exp = F.unfold(m(exp), [3, 3])
        up_exp = up_exp.view(N, 1, 9, 1, 1, H, W)

        up_exp = torch.sum(mask * up_exp, dim=2)
        up_exp = up_exp.permute(0, 1, 4, 2, 5, 3)
        return up_exp.reshape(N, 1, 8 * H, 8 * W)
    def upsample9_exp(self, exp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = exp.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_exp = exp.view(N, 1, 9, 1, 1, H, W)

        up_exp = torch.sum(mask * up_exp, dim=2)
        up_exp = up_exp.permute(0, 1, 4, 2, 5, 3)
        return up_exp.reshape(N, 1, 8 * H, 8 * W)
    def change_fun(self, exp):  # 将坐标转换成膨胀率
        exp = exp.clamp(0, 4)
        expo = torch.zeros_like(exp)
        expo[exp<=2] = exp[exp<=2] * 0.25 + 0.5
        expo[exp > 2] = 1/(1.5-exp[exp > 2]*0.25)
        return expo
    def normalize_features(self, feature_list, normalize, center, moments_across_channels=True, moments_across_images=True):
        """Normalizes feature tensors (e.g., before computing the cost volume).
        Args:
          feature_list: list of torch tensors, each with dimensions [b, c, h, w]
          normalize: bool flag, divide features by their standard deviation
          center: bool flag, subtract feature mean
          moments_across_channels: bool flag, compute mean and std across channels, 看到UFlow默认是True
          moments_across_images: bool flag, compute mean and std across images, 看到UFlow默认是True

        Returns:
          list, normalized feature_list
        """

        # Compute feature statistics.

        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = torch.mean(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            variance = torch.var(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)

        if moments_across_images:
            # statistics['mean'] = ([tf.reduce_mean(input_tensor=statistics['mean'])] *
            #                       len(feature_list))
            # statistics['var'] = [tf.reduce_mean(input_tensor=statistics['var'])
            #                      ] * len(feature_list)
            statistics['mean'] = ([torch.mean(torch.stack(statistics['mean'], dim=0), dim=(0,))] * len(feature_list))
            statistics['var'] = ([torch.var(torch.stack(statistics['var'], dim=0), dim=(0,))] * len(feature_list))

        statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

        # Center and normalize features.

        if center:
            feature_list = [
                f - mean for f, mean in zip(feature_list, statistics['mean'])
            ]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

        return feature_list
    def forward(self, image1, image2, iters=12, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        _, _, h, w = image1.shape

        image21 = F.conv2d(image2, self.weighto4, padding=2, groups=3)
        image21 = F.interpolate(image21, [int(h * self.delta4), int(w * self.delta4)])

        image23 = F.conv2d(image2, self.weighto2, padding=2, groups=3)
        image23 = F.interpolate(image23, [int(h * self.delta2), int(w * self.delta2)])

        image11 = F.conv2d(image1, self.weight14, padding=2, groups=3)
        image11 = F.interpolate(image11, [int(h * self.delta14), int(w * self.delta14)])

        image13 = F.conv2d(image1, self.weight12, padding=2, groups=3)
        image13 = F.interpolate(image13, [int(h * self.delta12), int(w * self.delta12)])
        #这里是反向的
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        Fmap1 = []
        Fmap2 = []

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):


            fmap2 = self.fnet(image21)
            fmap1 = self.fnet(image11)
            Fmap2.append(fmap2.float())
            Fmap1.append(fmap1.float())

            fmap2 = self.fnet(image23)
            fmap1 = self.fnet(image13)
            Fmap2.append(fmap2.float())
            Fmap1.append(fmap1.float())

            fmap2 = self.fnet(image2)
            fmap1 = self.fnet(image1)
            Fmap2.append(fmap2.float())
            Fmap1.append(fmap1.float())
            _, _, h1, w1 = Fmap2[2].shape



        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            #Initialize TPCV
            corr_fn = CorrpyBlock_3D_optical_flow(Fmap1, Fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet,centT = self.cnet([image1,image2])
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)


        coords0, coords1 = self.initialize_flow(image1)#初始化正向流
        exp = self.initialize_exp(image1)

        exp = exp.unsqueeze(1)

        flow_predictions = []
        exp_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            exp = exp.detach()

            #Calculate inverse optical flow
            flow2 = (coords1 - coords0)
            inv_coords = inverse_opticalflow(flow2, coords1, corr_fn.corr_pyramid[2])
            inv_coords[:, 0:2, :, :] = inv_coords[:, 0:2, :, :] + coords0
            idx_coords = inv_coords[:, 3:5, :, :]
            inv_coords = inv_coords[:,0:2,:,:]

            corr = corr_fn(coords1,exp,inv_coords,idx_coords)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask,scale_mask, delta_flow, dc_flow = self.update_block(net, inp, corr, flow, exp)


            coords1 = coords1 + delta_flow
            exp = exp + dc_flow
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                exp_up = self.upsample_exp(exp, scale_mask)
                exp_up = self.change_fun(exp_up)

            flow_predictions.append(flow_up)
            exp_predictions.append(exp_up)

        #Last update motion-in-depth (exp)
        exp = exp.detach()
        flow = (coords1 - coords0).detach()

        # Calculate inverse optical flow
        flow2 = (coords1 - coords0)
        inv_coords = inverse_opticalflow(flow2, coords1, corr_fn.corr_pyramid[2])
        inv_coords[:, 0:2, :, :] = inv_coords[:, 0:2, :, :] + coords0
        idx_coords = inv_coords[:, 3:5, :, :]
        inv_coords = inv_coords[:, 0:2, :, :]

        corr = corr_fn(coords1,exp,inv_coords,idx_coords)

        up_maskdc, dc_flowe = self.dc_block(net, inp, corr, flow, exp)
        exp = exp + dc_flowe * 0.005
        exp_up = self.upsample_exp(exp, up_maskdc)
        exp_up = self.change_fun(exp_up)
        exp_predictions.append(exp_up)

        if test_mode:
            return coords1 - coords0, flow_up, exp_up
        return flow_predictions, exp_predictions