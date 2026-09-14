"""Editör güvenilirlik metrikleri ve sayfa başına çatışma skoru."""


def compute_editor_features(revisions_df, as_of_timestamp=None):
    """Her (user, timestamp) çifti için o ana kadarki geçmişe dayalı
    editör feature'larını hesaplar: toplam edit sayısı, geçmiş revert oranı,
    ilk edit'ten bu yana geçen süre (hesap yaşı proxy'si).

    TODO
    """
    raise NotImplementedError


def compute_page_conflict_score(revisions_df):
    """Sayfa başına çatışma skoru: revert oranı, benzersiz editör sayısı,
    edit burst tespiti (kısa sürede yoğun düzenleme).

    TODO
    """
    raise NotImplementedError
