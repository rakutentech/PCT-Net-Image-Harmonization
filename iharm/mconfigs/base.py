from iharm.model.base import SSAMImageHarmonization, PCTNet
                    
BMCONFIGS = {

########################
###### ViT-based #######
########################

    'ViT_pct': {
            'model': PCTNet,
            'params': { 'backbone_type': 'ViT', 
                        'input_normalization': {'mean': [0,0,0], 'std':[1,1,1]},
                        'dim': 3, 'transform_type': 'linear', 'affine': True, 
                        'clamp': True, 'color_space': 'RGB', 'use_attn': False  
                        },
            'data': {'color_space': 'RGB'}
        },

    'ViT_pct_sym': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ViT', 
                    'input_normalization': {'mean': [0,0,0], 'std':[1,1,1]},
                    'dim': 3, 'transform_type': 'linear_sym', 'affine': False, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': False  
                    },
        'data': {'color_space': 'RGB'}
    },

    'ViT_pct_polynomial': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ViT', 
                    'input_normalization': {'mean': [0,0,0], 'std':[1,1,1]},
                    'dim': 3, 'transform_type': 'polynomial', 'affine': False, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': False  
                    },
        'data': {'color_space': 'RGB'}
    },

    'ViT_pct_identity': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ViT', 
                    'input_normalization': {'mean': [0,0,0], 'std':[1,1,1]},
                    'dim': 3, 'transform_type': 'identity', 'affine': False, 
                    'clamp': False, 'color_space': 'RGB', 'use_attn': False  
                    },
        'data': {'color_space': 'RGB'}
    },

    'ViT_pct_mul': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ViT', 
                    'input_normalization': {'mean': [0,0,0], 'std':[1,1,1]},
                    'dim': 3, 'transform_type': 'mul', 'affine': False, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': False  
                    },
        'data': {'color_space': 'RGB'}
    },

    'ViT_pct_add': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ViT', 
                    'input_normalization': {'mean': [0,0,0], 'std':[1,1,1]},
                    'dim': 3, 'transform_type': 'add', 'affine': False, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': False  
                    },
        'data': {'color_space': 'RGB'}
    },


########################
###### CNN-based #######
########################


    'CNN_pct': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ssam', 'depth': 4, 'ch': 32, 'image_fusion': True, 
                    'attention_mid_k': 0.5, 'batchnorm_from': 2, 'attend_from': 2,
                    'input_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                    'dim': 3, 'transform_type': 'linear', 'affine': True, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': True                     
                },
        'data': {'color_space': 'RGB'}
    },

    'CNN_pct_sym': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ssam', 'depth': 4, 'ch': 32, 'image_fusion': True, 
                    'attention_mid_k': 0.5, 'batchnorm_from': 2, 'attend_from': 2,
                    'input_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                    'dim': 3, 'transform_type': 'linear_sym', 'affine': True, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': True                     
                },
        'data': {'color_space': 'RGB'}
    },

    'CNN_pct_polynomial': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ssam', 'depth': 4, 'ch': 32, 'image_fusion': True, 
                    'attention_mid_k': 0.5, 'batchnorm_from': 2, 'attend_from': 2,
                    'input_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                    'dim': 3, 'transform_type': 'polynomial', 'affine': False, 
                    'clamp': True, 'color_space': 'RGB', 'use_attn': True                     
                },
        'data': {'color_space': 'RGB'}
    },

    'CNN_pct_identity': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ssam', 'depth': 4, 'ch': 32, 'image_fusion': True, 
                    'attention_mid_k': 0.5, 'batchnorm_from': 2, 'attend_from': 2,
                    'input_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                    'dim': 3, 'transform_type': 'identity', 'affine': False, 
                    'clamp': False, 'color_space': 'RGB', 'use_attn': True                     
                },
        'data': {'color_space': 'RGB'}
    },

    'CNN_pct_mul': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ssam', 'depth': 4, 'ch': 32, 'image_fusion': True, 
                    'attention_mid_k': 0.5, 'batchnorm_from': 2, 'attend_from': 2,
                    'input_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                    'dim': 3, 'transform_type': 'mul', 'affine': False, 
                    'clamp': False, 'color_space': 'RGB', 'use_attn': True                     
                },
        'data': {'color_space': 'RGB'}
    },

    'CNN_pct_add': {
        'model': PCTNet,
        'params': { 'backbone_type': 'ssam', 'depth': 4, 'ch': 32, 'image_fusion': True, 
                    'attention_mid_k': 0.5, 'batchnorm_from': 2, 'attend_from': 2,
                    'input_normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                    'dim': 3, 'transform_type': 'add', 'affine': False, 
                    'clamp': False, 'color_space': 'RGB', 'use_attn': True                     
                },
        'data': {'color_space': 'RGB'}
    },

}