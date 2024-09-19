import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.ops import deform_conv2d
from torchvision.models import resnet34
import torch.nn.functional as F

class VGG16_Separate(nn.Module):
    def __init__(self):
        super(VGG16_Separate,self).__init__()

        vgg_model = models.vgg16(pretrained=True)
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])

    def forward(self,x):

        out1 = self.Conv1(x)

        out2 = self.Conv2(out1)

        out3 = self.Conv3(out2)

        out4 = self.Conv4(out3)

        out5 = self.Conv5(out4)


        return out1, out2, out3, out4, out5
        
                
class ResNet50_Separate(nn.Module):
    def __init__(self):
        super(ResNet50_Separate,self).__init__()
        
        resnet = models.resnet50(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        
    def forward(self, x):
         
        out1 = self.firstrelu(self.firstbn(self.firstconv(x)))
        out11 = self.firstmaxpool(out1)
        out2 = self.encoder1(out11)
        out3 = self.encoder2(out2)
        out4 = self.encoder3(out3)
        out5 = self.encoder4(out4)       
        
        return out1, out2, out3, out4, out5  


class ResNet34_Separate(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34_Separate,self).__init__()
        resnet = resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
                
                
    def forward(self,x):

        x = self.firstconv(x)
        x = self.firstbn(x)
        c1 = self.firstrelu(x)#1/2  64
        x = self.firstmaxpool(c1)
        
        c2 = self.encoder1(x)#1/4   64
        c3 = self.encoder2(c2)#1/8   128
        c4 = self.encoder3(c3)#1/16   256
        c5 = self.encoder4(c4)#1/32   512
        
        return c1, c2, c3, c4, c5        
           

class PoolUp(nn.Module):
      def __init__(self, input_channels, pool_kernel_size, reduced_channels):
          super().__init__()

          self.pool = nn.AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d(kernel_size=pool_kernel_size, stride = pool_kernel_size)
          self.up = nn.Upsample(scale_factor=pool_kernel_size)
          
      def forward(self,x):

          y = self.pool(x)
          y = self.up(y)
          
          return y


class Pool_pixelWise(nn.Module):
      def __init__(self, input_channels, pool_kernel_size, reduced_channels):
          super().__init__()

          self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride = 1, padding = pool_kernel_size//2)
          self.conv = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, padding=0)
          
          
      def forward(self,x):

          y = self.pool(x)
          y = self.conv(y)
          
          return y



class PVF(nn.Module):
    def __init__(self, input_channels, pool_sizes, dim):
        super().__init__()
        
        reduced_channels = input_channels//8
        output_channels = input_channels//2
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)

        self.PoolUp1 = PoolUp(output_channels, pool_sizes[0], reduced_channels)
        self.PoolUp2 = nn.AvgPool2d(kernel_size=pool_sizes[1], padding = (pool_sizes[1])//2, stride=1)
        self.PoolUp3 = nn.AvgPool2d(kernel_size=pool_sizes[2], padding = (pool_sizes[2])//2, stride=1)
        self.PoolUp4 = nn.AvgPool2d(kernel_size=pool_sizes[3], padding = (pool_sizes[3])//2, stride=1)

        self.conv1 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        
        self.fc = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)
        self.norm = nn.LayerNorm([dim,dim])

        
    def forward(self,x):


        y = self.conv(x)
        
        glob1 = self.conv1(self.PoolUp1(y))       
        glob2 = self.conv2(self.PoolUp2(y))       
        glob3 = self.conv3(self.PoolUp3(y))       
        glob4 = self.conv4(self.PoolUp4(y))
     
        concat = torch.cat([y,glob1,glob2,glob3,glob4],1)       
        fully_connected = self.fc(concat)       
        z = self.norm(fully_connected)
        
        return z




class Up(nn.Module):

    def __init__(self, in_channels, out_channels, dilations, norm_shape, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DPR(in_channels, out_channels, dilations, norm_shape)

        else:
            raise Exception("Upscaling with other schemes rather than bilinear is not implemented")

            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)            


    
class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilate):
        super().__init__()
        
        self.offset = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=(2*(dilate+1)+1), padding = dilate+1, dilation = 1)
        self.tan = nn.Hardtanh()
        self.dilate = dilate
        
    def forward(self,x, weights):
        
        off = self.offset(x)
        off1 = self.tan(off)

        out = deform_conv2d(input=x, offset=off1, weight=weights, stride = (1,1), 
                                    padding = self.dilate, dilation = self.dilate)

        return out


class DPR(nn.Module):

    def __init__(self, in_channels, out_channels, dilations, norm_shape):
        super().__init__()
           
        self.conv0 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv1 = DeformableBlock(in_channels, in_channels//2, kernel_size=3, dilate = dilations[0])
        self.conv2 = DeformableBlock(in_channels, in_channels//2, kernel_size=3, dilate = dilations[1])

        self.evaluator = nn.Conv2d(in_channels//2, 1, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        
        self.out = nn.Sequential(
                   nn.LayerNorm([in_channels//2, norm_shape[0], norm_shape[1]], elementwise_affine=False),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1),
                   nn.LayerNorm([out_channels, norm_shape[0], norm_shape[1]], elementwise_affine=False),
                   nn.ReLU(inplace=True))
   

    def forward(self, x):
        
        x0 = self.conv0(x)
        x1 = self.conv1(x, weights=self.conv0.weight)
        x2 = self.conv2(x, weights=self.conv0.weight)

        # print(f'x0.shape: {x0.shape}')

        x00 = self.evaluator(x0)
        x11 = F.conv2d(x1, weight=self.evaluator.weight)
        x22 = F.conv2d(x2, weight=self.evaluator.weight)

        # print(f'x22.shape: {x22.shape}')
        
        y = self.softmax(torch.cat([x00, x11, x22], dim=1))

        # print(f'y.shape: {y.shape}')

        y1 = torch.mul(y[:,0,:,:].unsqueeze(-3),x0) + torch.mul(y[:,1,:,:].unsqueeze(-3),x1) + torch.mul(y[:,2,:,:].unsqueeze(-3),x2)

        y2 = self.out(y1)
        
        return y2
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)    



if __name__ == '__main__':

    template = torch.ones((1, 3, 512, 512))
    
    model =VGG16_Separate()
    y = model(template)
    print(f'Output Shape: {y.shape}')

    

