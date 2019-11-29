module Main where

import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Control.Monad.Trans.State
import Graphics.Gloss.Interface.IO.Simulate
import Graphics.Gloss.Data.ViewPort
import RL.Gym

data IState g = IState { gym :: g
                       , obs :: Observation g
                       , lr  ::Bool
                       }

istate :: (Monad m) => (g -> (a,g)) -> StateT (IState g) m a
istate f = do
    (out, ngym)  <- fmap (f . gym) get
    modify $ \st -> st{gym = ngym}
    pure out

defSt :: Gym g => IO (IState g)
defSt = fmap (\(o, g) -> IState g o False) . fmap reset $ start

rnd :: Gym g => IState g -> IO Picture
rnd = pure . translate (-300) (-200) . render 600 400 . gym

stp :: ViewPort -> Float -> IState Cartpole -> IO (IState Cartpole)
stp _ _ = execStateT $ do
    act <- fmap (\s -> if lr s then CLeft else CRight) get
    (o, r, d) <- istate (step act)
    o <- if d then liftIO (putStrLn "Reset") >> istate reset else pure o
    modify (\s -> s{obs = o})

main :: IO ()
main = do
    st <- defSt
    simulateIO (InWindow "Cartpole" (600,400) (10,10)) (makeColor 1 1 1 1) 50 st rnd stp
