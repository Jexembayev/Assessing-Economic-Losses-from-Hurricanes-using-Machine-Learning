class SiameseUnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        unet_channels: List[int],
        dropout=0.25,
        abn_block: Union[ABN, Callable[[int], nn.Module]] = ABN,
        full_size_mask=True,
        activation=ACT_RELU,
        upsample_block=nn.UpsamplingNearest2d,
    ):
        super().__init__()
        self.encoder = encoder

        feature_maps = [2 * fm for fm in encoder.channels]
        
        abn_block = partial(ABN, activation=activation)
        self.decoder = UNetDecoder(
            feature_maps=feature_maps,
            decoder_features=unet_channels,
            unet_block=partial(UnetBlock, abn_block=abn_block),
            upsample_block=upsample_block,
        )

        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(unet_channels[0], num_classes))])
        )        

    def forward(self, image):
        batch_size = image.size(0)
        # Split input image batch into pre- and post- batches
        pre, post = image[:, 0:3, ...], image[:, 3:6, ...] # [B,3,H,W], [B,3,H,W]
        
        # Concatenate them along batch dimension since it's faster than calling self.encoder(pre), self.encoder(post)
        x = torch.cat([pre, post], dim=0) # [2 * B, 3, H, W]
        
        # Encoder
        features = self.encoder(x) # List[Tensor] of strides 4,8,16,32
        
        pre_features = [f[0: batch_size] for f in features]
        post_features = [f[batch_size: batch_size * 2] for f in features]
        features = [torch.cat([pre, post], dim=1) for pre, post in zip(pre_features, post_features)]
        
        # Decoder part
        features = self.decoder(features)
        
        # Decode mask
        mask = self.mask(features[0])
        mask = F.interpolate(mask, size=image.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_DAMAGE_MASK_KEY: mask}
        return output
def siamese_b4_unet64(input_channels=3, num_classes=5, dropout=0.2, pretrained=True):
    encoder = B4Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return SiameseUnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[64, 128, 256], activation=ACT_SWISH, dropout=dropout
    )
SIAMESE_BATCH_SIZE = 8
best_checkpoint_fname, best_f1 = train_closure(
    model=siamese_b4_unet64().cuda(),
    batch_size=SIAMESE_BATCH_SIZE,
    experiment_name="b4-siamese")
model = siamese_b4_unet64().cuda()
model.load_state_dict(torch.load(best_checkpoint_fname)["model_state_dict"])

for i in range(len(holdout_mask_post)):
    gt_mask = read_mask(holdout_mask_post[i])
    image_pre = read_image_rgb(holdout_img_pre[i])
    image_post = read_image_rgb(holdout_img_post[i])
    pred_mask = predict(model, image_pre, image_post)
    
    show_predictions(image_post, gt_mask, pred_mask)
    
del model
@torch.no_grad()
def predict(model, image_pre, image_post):
    if isinstance(image_pre, str):
        image_pre = read_image_rgb(image_pre)
    if isinstance(image_post, str):
        image_post = read_image_rgb(image_post)
        
    normalize = A.Normalize()
    image = np.dstack([
        normalize(image=image_pre)["image"],
        normalize(image=image_post)["image"],
    ])
    model_input = image_to_tensor(image).unsqueeze(0).cuda()
    outputs = model.eval()(model_input)
    
    mask = outputs[OUTPUT_DAMAGE_MASK_KEY].argmax(dim=1)
    return to_numpy(mask[0])

def show_predictions(image, gt_mask, pred_mask):
    f, ax = plt.subplots(1, 3, figsize=(30,10))
    
    gt_overlay = overlay_damage_mask(image, gt_mask)
    pred_overlay = overlay_damage_mask(image, pred_mask)
    
    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].imshow(gt_overlay)
    ax[1].axis('off')

    ax[2].imshow(pred_overlay)
    ax[2].axis('off')
    
    f.tight_layout()
    f.show()
