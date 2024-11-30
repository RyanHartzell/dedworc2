# import pygame
# from pygame._sdl2 import Window, Texture, Renderer
# pygame.init() 

# window_1 = Window("Window 1", size=(300, 200))  # create windows 
# window_2 = Window("Window 2", size=(300,200), position(window_1.position[0] + 350, window_1.position[1]))

# window_1.show() 
# window_2.show() 

# render_1 = Renderer(window_1)  # create render context for windows 
# render_2 = Renderer(window_2)

# #example surface

# surf = pygame.Surface([50, 50]) 
# surf.fill((255, 255, 255))

# texture = Texture.from_surface(render_1, surf)
# surf.fill((255, 0, 0))
# texture_2 = Texture.from_surface(render_2, surf)

# running = True 
# while running: 
#     render_1.clear()  # clears the window render_2.clear() 
#     for event in pygame.event.get(): 
#         if event.type == pygame.QUIT: 
#             running = False
#         elif getattr(event, "window", None) == window_1:
#             if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE or event.type == pygame.WINDOWCLOSE:
#                 window_1.destroy()
    
#         elif getattr(event, "window", None) == window_2:
#             if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE or event.type == pygame.WINDOWCLOSE:
#                 window_2.destroy()
    
#     texture.draw(dstrect=(10, 10))  # draws the texture (dstrect is neccessary )
#     texture_2.draw(dstrect=(10, 10))
    
#     render_1.present()  # similar to pygame.display.flip()
#     render_2.present()