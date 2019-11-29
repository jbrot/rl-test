module Main where

import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Graphics.Gloss
import Graphics.Gloss.Data.ViewPort
import RL.Gym

type IState = Bool

defSt :: IState
defSt = False

rnd :: (Cartpole, IState) -> Picture
rnd = translate (-300) (-200) . render 600 400 . fst

stp :: ViewPort -> Float -> (Cartpole, IState) -> (Cartpole, IState)
stp _ _ (c, s) = (fst (step act c), not s)
    where act = if s then CLeft else CRight

main :: IO ()
main = do
    st <- start
    simulate (InWindow "Cartpole" (600,400) (10,10)) (makeColor 1 1 1 1) 50 (st, defSt) rnd stp
