{-# LANGUAGE MultiParamTypeClasses, TypeFamilies #-}
module RL.Gym.Types where

import Data.Word
import Graphics.Gloss (Picture)


class Gym t where
    data Action t :: *
    data Observation t :: *
    step :: Action t -> t -> ((Observation t, Float, Bool), t)
    reset :: t -> (Observation t, t)
    render :: Float -> Float -> t -> Picture -- Width -> Height -> State -> Picture
    seed :: Maybe Word64 -> t -> IO ((), t)
    start :: IO t
