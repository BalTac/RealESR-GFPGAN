import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def create_banner(output_path="assets/banner.png", width=1280, height=420):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Base image with deep slate/indigo gradient
    img = Image.new("RGBA", (width, height), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        ratio = y / height
        # Gradient from #0b0f19 (11, 15, 25) to #161a35 (22, 26, 53)
        r = int(11 + ratio * 15)
        g = int(15 + ratio * 12)
        b = int(25 + ratio * 35)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
        
    # 2. Glowing radial ambient highlight in background (left & right)
    glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    
    # Cyan glow on bottom left
    for radius in range(250, 0, -5):
        alpha = int((1 - radius / 250) * 45)
        glow_draw.ellipse(
            [120 - radius, 280 - radius, 120 + radius, 280 + radius],
            fill=(6, 182, 212, alpha)
        )
    # Violet glow on top right
    for radius in range(350, 0, -8):
        alpha = int((1 - radius / 350) * 55)
        glow_draw.ellipse(
            [1050 - radius, 140 - radius, 1050 + radius, 140 + radius],
            fill=(139, 92, 246, alpha)
        )
        
    img = Image.alpha_composite(img, glow_layer)
    draw = ImageDraw.Draw(img)
    
    # 3. Subtle neural grid lines
    grid_color = (255, 255, 255, 12)
    grid_step = 40
    for x in range(0, width, grid_step):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_step):
        draw.line([(y, 0), (y, height)], fill=grid_color, width=1)
        
    # 4. Neural node network illustration on the right side
    net_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    net_draw = ImageDraw.Draw(net_layer)
    
    nodes = [
        # Layer 1: Inputs
        (900, 110), (900, 200), (900, 290),
        # Layer 2: RRDB & Prior Features
        (990, 80), (990, 150), (990, 240), (990, 320),
        # Layer 3: Blending Latents
        (1080, 120), (1080, 200), (1080, 280),
        # Layer 4: Output Synthesis
        (1170, 160), (1170, 240)
    ]
    
    connections = [
        (0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6),
        (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9),
        (7, 10), (8, 10), (8, 11), (9, 11)
    ]
    
    for start_idx, end_idx in connections:
        p1 = nodes[start_idx]
        p2 = nodes[end_idx]
        net_draw.line([p1, p2], fill=(139, 92, 246, 75), width=2)
        
    for x, y in nodes:
        # Glow around nodes
        net_draw.ellipse([x-8, y-8, x+8, y+8], fill=(6, 182, 212, 60))
        net_draw.ellipse([x-4, y-4, x+4, y+4], fill=(224, 242, 254, 230))
        
    img = Image.alpha_composite(img, net_layer)
    draw = ImageDraw.Draw(img)
    
    # 5. Accent top gradient bar
    for x in range(width):
        t = x / width
        # Cyan (6, 182, 212) -> Violet (139, 92, 246) -> Pink (236, 72, 153)
        if t < 0.5:
            s = t / 0.5
            r = int(6 + s * (139 - 6))
            g = int(182 + s * (92 - 182))
            b = int(212 + s * (246 - 212))
        else:
            s = (t - 0.5) / 0.5
            r = int(139 + s * (236 - 139))
            g = int(92 + s * (72 - 92))
            b = int(246 + s * (153 - 246))
        draw.line([(x, 0), (x, 4)], fill=(r, g, b, 255))
        draw.line([(x, height - 2), (x, height)], fill=(r, g, b, 120))
        
    # 6. Typography & Fonts
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 54)
        font_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
        font_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 13)
        font_tag = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 12)
    except:
        font_title = font_sub = font_badge = font_tag = ImageFont.load_default()
        
    # Category tag
    tag_text = "DEEP LEARNING IMAGE RESTORATION • 2026"
    draw.text((72, 65), tag_text, font=font_tag, fill=(6, 182, 212, 240))
    
    # Title with subtle shadow
    title_text = "RealESR-GFPGAN"
    draw.text((74, 98), title_text, font=font_title, fill=(10, 14, 23, 200))
    draw.text((72, 96), title_text, font=font_title, fill=(255, 255, 255, 255))
    
    # Subtitle
    sub_text = "Unified Photorealistic Face & Scene Super-Resolution"
    draw.text((72, 172), sub_text, font=font_sub, fill=(148, 163, 184, 255))
    
    # Description line
    desc_text = "Harmonizing 23-Block RRDBNet continuous backgrounds with calibrated generative face priors."
    draw.text((72, 210), desc_text, font=font_badge, fill=(100, 116, 139, 255))
    
    # 7. Badges / Feature Pills
    badges = [
        ("⚡ 23-Block RRDBNet Engine", (6, 182, 212)),
        ("🎯 Latent Prior Calibration", (139, 92, 246)),
        ("✨ Seamless Alpha Blend", (236, 72, 153)),
        ("📦 Single .pth Checkpoint", (34, 197, 94))
    ]
    
    bx = 72
    by = 275
    badge_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(badge_layer)
    
    for text, (cr, cg, cb) in badges:
        bbox = font_badge.getbbox(text)
        bw = (bbox[2] - bbox[0]) + 26
        bh = 34
        
        # Pill background
        bdraw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=8, fill=(cr, cg, cb, 28), outline=(cr, cg, cb, 90), width=1)
        # Pill text
        bdraw.text((bx + 13, by + 8), text, font=font_badge, fill=(255, 255, 255, 230))
        bx += bw + 14
        
    img = Image.alpha_composite(img, badge_layer)
    
    # Save optimized PNG
    img.save(output_path, "PNG", optimize=True)
    print(f"Banner generated successfully at {output_path} ({width}x{height})")

if __name__ == "__main__":
    create_banner()
